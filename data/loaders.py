import math
import os
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from torchvision import transforms as T
import yt_dlp

from data.pre_process import LetterBox
from utils.logger import LOGGER


def get_best_youtube_url(url: str) -> Optional[str]:
    """
    Retrieves the best quality MP4 video stream URL from a given YouTube video.

    Uses either the pafy or yt_dlp library to extract video information from YouTube. Finds the highest quality MP4
    format that has a video codec but no audio codec, and returns the URL of this video stream.

    Parameters:
    -----------
    url : str
        The URL of the YouTube video.

    Returns:
    --------
    Optional[str]
        The URL of the best quality MP4 video stream, or None if no suitable stream is found.
    """
    try:
        import pafy
        return pafy.new(url).getbest(preftype='mp4').url
    except ImportError:
        LOGGER.warning('WARNING ⚠️ pafy library not found, falling back to yt_dlp.')
        use_pafy = False
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info_dict = ydl.extract_info(url, download=False)  # Extract video info
    for f in info_dict.get('formats', None):
        if f['vcodec'] != 'none' and f['acodec'] == 'none' and f['ext'] == 'mp4':
            return f.get('url', None)


class LoadStreams:
    """
    Load video streams for predictions. Supports various video sources including files, RTSP, RTMP, HTTP streams, 
    and YouTube links.
    
    Attributes:
    -----------
    mode : str
        Operating mode, set to 'stream'.
    imgsz : Tuple[int, int]
        Shape to resize the image to.
    vid_stride : int
        Video frame-rate stride.
    sources : List[str]
        List of source video links or paths.
    LB : Type[LetterBox]
        A LetterBox object for resizing and padding.
    imgs : List[Union[np.ndarray, None]]
        List of images from video sources.
    fps : List[int]
        List of frames per second for each video source.
    frames : List[Union[int, float]]
        List of total frames for each video source.
    threads : List[Union[Thread, None]]
        List of threads for each video source.
    bs : int
        Batch size, determined by the number of sources.
    count : int
        Frame count for iteration.
    verbose : bool
        If True, provides detailed logging.
    """
    
    def __init__(self, sources: str = 'file.streams', imgsz: Union[int, Tuple[int, int]] = 640, 
                 vid_stride: int = 1, verbose: bool = False):
        """
        Initialize LoadStreams object with specific parameters.
        
        Parameters:
        -----------
        sources : str, default='file.streams'
            The source file path or single source URL.
        imgsz : Union[int, Tuple[int, int]], default=(640, 640)
            Shape to resize image to.
        vid_stride : int, default=1
            Video frame-rate stride.
        verbose : bool, default=False
            If True, provides detailed logging.
        """
        torch.backends.cudnn.benchmark = True  # For faster fixed-size inference
        self.mode = 'stream'
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.vid_stride = vid_stride
        self.verbose = verbose
        sources = Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
        self.LB = LetterBox(self.imgsz)
        self.transform = T.Compose([
            # lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        n = len(sources)
        self.sources = [str(x) for x in sources]  # Clean source names for later use
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        # Initialize video streams and start reading frames
        for i, s in enumerate(sources):
            if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # YouTube source
                s = get_best_youtube_url(s)
            s = eval(s) if s.isnumeric() else s  # Local webcam if s = '0'
            cap = cv2.VideoCapture(s)
            if not cap.isOpened():
                raise ConnectionError(f'Failed to open {s}')
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # Fallback to infinite stream
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 60  # Fallback to 60 FPS
            success, self.imgs[i] = cap.read()  # Guarantee first frame
            if not success or self.imgs[i] is None:
                raise ConnectionError(f'Failed to read images from {s}')
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f'Success ✅ ({self.frames[i]} frames of shape {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')  # Newline for formatting
        self.bs = self.__len__()  # Batch size

    def update(self, i: int, cap: cv2.VideoCapture, stream: str):
        """
        Read stream `i` frames in daemon thread.
        
        Parameters:
        -----------
        i : int
            Index of the stream to update.
        cap : cv2.VideoCapture
            Video capture object for the stream.
        stream : str
            Source URL or path of the video stream.
        """
        n, f = 0, self.frames[i]
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % self.vid_stride == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # Re-open stream if signal is lost

    def __iter__(self):
        """
        Returns iterator for image feed and re-opens unresponsive streams.
        
        Returns:
        --------
        self
        """
        self.count = -1
        return self # return self.__next__()

    def __next__(self):
        """
        Returns source paths, transformed and original images for processing.
        
        Returns:
        --------
        Tuple[List[str], List[np.ndarray], torch.Tensor, str]
            Source paths, original images, transformed images, and an empty string.
        """
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        img0 = self.imgs.copy()
        img = [self.LB(image=x) for x in img0]  # Apply LetterBox transformation
        img = np.stack(img)
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img).astype(np.float32)
        img = torch.from_numpy(img)/255.0
        # assert img.shape == torch.Size([self.bs, 3, *self.imgsz]), f'Image size {img.shape} not consistent with batch size {self.bs}'
        img = self.transform(img).cuda()
        return self.sources, img0, img, ''

    def __len__(self) -> int:
        """
        Returns the length of the sources object.
        
        Returns:
        --------
        int
            The length of the sources object.
        """
        return len(self.sources)
    


       
