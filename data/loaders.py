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

class BaseDataLoader:
    """
    Base class for data loading. 
    
    Attributes:
    -----------
    mode : str
        Operating mode, can be 'stream', 'batch', etc.
    sources : List[str]
        List of source links or paths.
    imgs : List[Union[np.ndarray, None]]
        Image buffer from data sources (BGR).
    threads : List[Union[Thread, None]]
        List of threads for each data source.
    bs : int
        Batch size, determined by the number of sources.
    count : int
        Frame count for iteration.
    verbose : bool
        If True, provides detailed logging.
    """
    
    def __init__(self, sources: List[str], imgsz: Union[int, Tuple[int, int]], device:torch.device=torch.device("cuda:0"), verbose: bool = False):
        """
        Initialize BaseDataLoader object with specific parameters.
        
        Parameters:
        -----------
        sources : List[str]
            List of source file paths or URLs.
        verbose : bool, default=False
            If True, provides detailed logging.
        """
        self.mode = None  # To be defined by subclass
        self.sources = sources
        self.imgs = [None] * len(sources)
        self.threads = [None] * len(sources)
        self.bs = len(sources)
        self.count = -1
        self.verbose = verbose
        self.transform = lambda x: x  # To be defined by subclass
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.device = device

    def _preprocess(self, img0) -> torch.Tensor:
        """
        Preprocesses images or frames.

        Parameters:
        -----------
        img0 : List[np.ndarray]
            List of images or frames to preprocess.
        
        Returns:
        --------
        torch.Tensor
            Preprocessed images or frames (BCHW). Batch size is determined by the number of sources.
        """
        img = [self.LB(image=x) for x in img0]  # Apply LetterBox transformation
        img = np.stack(img)
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img).astype(np.float32)
        img = torch.from_numpy(img)/255.0
        img = self.transform(img).to(self.device)
        return img

        
    def update(self, i: int, cap: Union[None, cv2.VideoCapture], stream: str):
        """
        Abstract method for updating data stream in a thread.
        
        Parameters:
        -----------
        i : int
            Index of the stream to update.
        cap : Union[None, cv2.VideoCapture]
            Capture object for the stream (if applicable).
        stream : str
            Source URL or path of the data stream.
        """
        raise NotImplementedError("This method must be overridden by the subclass.")
        
    def __iter__(self):
        """
        Returns iterator for data feed.
        
        Returns:
        --------
        self
        """
        self.count = -1
        return self

    def __next__(self):
        """
        Abstract method for obtaining the next batch of data.
        
        Returns:
        --------
        Tuple
            Defined by subclass.
        """
        self._check_threads()
        img0 = self.imgs.copy()
        img = self._preprocess(img0)
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
    
class DataLoaderVideo(BaseDataLoader):
    """
    DataLoader for video streams.
    
    Inherits from:
    --------------
    BaseDataLoader
    
    Methods:
    --------
    __init__ : Initialize DataLoaderVideo object with specific parameters.
    init_attributes : Initialize attributes specific to video streams.
    init_transforms : Initialize image transformations.
    init_streams : Initialize video streams.
    init_single_stream : Initialize a single video stream.
    update : Read stream frames in daemon thread.
    _check_threads : Check if threads are alive and re-open unresponsive streams.
    
    Attributes:
    -----------
    mode : str
        Operating mode, set to 'stream'.
    vid_stride : int
        Frame stride for video streams.
    
    Note:
    -----
    This class is designed for video stream data loading.
    """
    def __init__(self, sources: str = 'file.streams', imgsz: Union[int, Tuple[int, int]] = 640, 
                 vid_stride: int = 1, **kwargs):
        """
        Initialize DataLoaderVideo object with specific parameters.
        
        Parameters:
        -----------
        sources : str, default='file.streams'
            Source file paths or URLs for video streams.
        imgsz : Union[int, Tuple[int, int]], default=640
            Image size for video frames.
        vid_stride : int, default=1
            Frame stride for video streams.
        verbose : bool, default=False
            If True, provides detailed logging.
        """
        
        # Initialize parent class
        super().__init__(sources=[sources], imgsz=imgsz, **kwargs)
        
        # Additional LoadStreams-specific initializations
        self.init_attributes(vid_stride)
        self.init_transforms()
        self.init_streams()

    def init_attributes(self, vid_stride):
        """
        Initialize attributes specific to video streams.
        
        Parameters:
        -----------
        vid_stride : int
            Frame stride for video streams.
        
        Note:
        -----
        Sets the mode to 'stream' and initializes CUDA benchmarking.
        """
        self.mode = 'stream'
        self.vid_stride = vid_stride
        torch.backends.cudnn.benchmark = True  # For faster fixed-size inference

    def init_transforms(self)->None:
        """
        Initialize image transformations.
        
        Note:
        -----
        Initializes the LetterBox transformation and normalization.
        """
        self.LB = LetterBox(self.imgsz)
        self.transform = T.Compose([
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            T.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])

    def init_streams(self)->None:
        """
        Initialize video streams.
        
        Note:
        -----
        Initializes video streams based on the provided source files or URLs.
        """
        sources = Path(self.sources[0]).read_text().rsplit() if os.path.isfile(self.sources[0]) else self.sources
        self.sources = [str(x) for x in sources]
        self.imgs, self.fps, self.frames, self.threads = [None] * len(self.sources), [0] * len(self.sources), [0] * len(self.sources), [None] * len(self.sources)
        
        for i, s in enumerate(self.sources):
            cap = self.init_single_stream(i, s)
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            self.threads[i].start()

    def init_single_stream(self, i, s)->cv2.VideoCapture:
        """
        Initialize a single video stream.
        
        Parameters:
        -----------
        i : int
            Index of the stream to initialize.
        s : str
            Source URL or path for the video stream.
        
        Returns:
        --------
        cv2.VideoCapture
            Initialized video capture object.
        
        Raises:
        -------
        ConnectionError
            When the video stream fails to open or read.
        
        Note:
        -----
        Handles YouTube sources and local webcams.
        """
        # YouTube or local webcam handling here
        if urlparse(s).hostname in ('www.youtube.com', 'youtube.com', 'youtu.be'):  # YouTube source
            s = get_best_youtube_url(s)
        s = eval(s) if s.isnumeric() else s  # Local webcam if s = '0'

        # Initialize video stream
        cap = cv2.VideoCapture(s)
        if not cap.isOpened():
            raise ConnectionError(f'Failed to open {s}')

        w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))//self.vid_stride, 0) or float('inf')  # Fallback to infinite stream
        self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 60  # Fallback to 60 FPS

        # Read the first frame to ensure the stream is functioning
        success, self.imgs[i] = cap.read()
        if not success or self.imgs[i] is None:
            raise ConnectionError(f'Failed to read images from {s}')

        # Logging
        LOGGER.info(f'Success ✅ ({self.frames[i]} frames of {w}x{h} at {self.fps[i]:.2f} FPS)')
        return cap
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
        
        Note:
        -----
        Reads frames from the video stream and updates the image buffer (self.imgs).
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
    def _check_threads(self)->None:
        """
        Check if threads are alive and re-open unresponsive streams.
        
        Note:
        -----
        Checks if threads are alive and re-opens unresponsive streams.
        """
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration