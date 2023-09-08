import glob
import math
import os
from multiprocessing import Pool
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse

import cv2
import imageio
import numpy as np
import torch
import yt_dlp
from torchvision import transforms as T

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
        self.sources = sources if isinstance(sources, list) else [sources]
        self.meta = None
        self.imgs = [None] * len(sources)
        self.threads = [None] * len(sources)
        self.frames = [0] * len(sources)
        self.bs = len(sources)
        self.count = -1
        self.verbose = verbose
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.device = device
        self.init_transforms()

    def _preprocess(self, img0:Union[list,np.ndarray]) -> torch.Tensor:
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
        assert isinstance(img0, list) or isinstance(img0,np.ndarray), f'img0 must be a list or np.ndarray. Got {type(img0)}'
        img = np.array([self.LB(image=x) for x in img0])  # Apply LetterBox transformation
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img).astype(np.float32)
        img = torch.from_numpy(img)
        img = self.transform(img).to(self.device)
        return img
    
    def init_transforms(self)->None:
        """
        Initialize image transformations.
        
        Note:
        -----
        Initializes the LetterBox transformation and normalization.
        """
        assert isinstance(self.imgsz, tuple), f'imgsz must be a tuple. Got {type(self.imgsz)}'
        self.LB = LetterBox(self.imgsz)
        self.transform = T.Compose([
            # T.Lambda(lambda x: 255.0 * x[:3]), # Discard alpha component and scale by 255
            T.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])
    def pre_hooks(self)->None:
        """
        Pre-iteration hooks.
        
        Note:
        -----
        Currently does nothing. To be defined by subclass.
        """
        pass
        
    def _check_alive(self)->bool:
        """
        Abstract method for checking if it should continue iterating.
        
        Returns:
        --------
        bool
            True if it should continue iterating, False otherwise.
        """
        return self.count < self.frames
    
    def __iter__(self):
        """
        Returns iterator for data feed.
        
        Returns:
        --------
        self
        """
        self.count -= 1
        return self

    def __next__(self)->Tuple[List[str], List[np.ndarray], torch.Tensor, str]:
        """
        Abstract method for obtaining the next batch of data.
        
        Returns:
        --------
        Tuple
            Defined by subclass.
        """
        self.count += 1

        self._check_alive()
        self.pre_hooks()
        img0 = self.imgs.copy()
        img = self._preprocess(img0)
        return self.sources, img0, img, self.meta
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Returns the next batch of data.
    
        Returns:
        --------
        Tuple
            Defined by subclass.
        """
        return next(self)
           
        
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
    def _check_alive(self)->None:
        """
        Check if threads are alive and re-open unresponsive streams.
        
        Note:
        -----
        Checks if threads are alive and re-opens unresponsive streams.
        """
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        

class DataLoaderImages(BaseDataLoader):
    """
    DataLoader for image files.
    
    Inherits from:
    --------------
    BaseDataLoader
    
    Methods:
    --------
    __init__ : Initialize DataLoaderImages object with specific parameters.
    init_images : Initialize image files or directory.
    read_image : Read image using imageio.
    
    Attributes:
    -----------
    mode : str
        Operating mode, set to 'batch'.
    images : np.ndarray
        Numpy array storing all images.
    """
    def __init__(self, sources: List[str], imgsz: Union[int, Tuple[int, int]] = 640, bs:int=1, **kwargs):
        """
        Initialize DataLoaderImages object with specific parameters.
        
        Parameters:
        -----------
        sources : List[str]
            List of source file paths or directories for image files.
        imgsz : Union[int, Tuple[int, int]], default=640
            Image size for frames.
        verbose : bool, default=False
            If True, provides detailed logging.
        """
        
        # Initialize parent class
        super().__init__(sources=sources, imgsz=imgsz, **kwargs)
        
        # DataLoaderImages-specific initializations
        self.mode = 'batch'
        self.bs = bs
        self.init_images()

    def read_image(self,filename)->np.ndarray:
        """
        Read image using imageio, used for multiprocessing.
        
        Parameters:
        -----------
        filename : str
            File path for the image.
            
        Returns:
        --------
        np.ndarray
            Image as a numpy array.
        """
        return imageio.imread(filename)

    def init_images(self)->None:
        """
        Initialize image files.
        
        Note:
        -----
        Checks if image files exist based on the provided source file paths.
        """
        new_sources = []
        for src in self.sources:
            if os.path.isdir(src):
                new_sources.extend(glob.glob(os.path.join(src, '*.[jJ][pP][gG]')))
                new_sources.extend(glob.glob(os.path.join(src, '*.[pP][nN][gG]')))
                new_sources.extend(glob.glob(os.path.join(src, '*.[jJ][pP][eE][gG]')))
            else:
                new_sources.append(src)
        
        self.sources = np.array([str(x) for x in new_sources])
        
        # Check if images exist
        for s in self.sources:
            if not os.path.exists(s):
                raise FileNotFoundError(f'Failed to find {s}')

        # Read images into a numpy array
        with Pool() as p:
            self.images = p.map(self.read_image, self.sources)
        self.images = np.stack(self.images, axis=0)
    def pre_hooks(self) -> None:
        """
        Select the next batch of images.
        """
        self.imgs = [None] * self.bs
        indices = np.arange(self.count, self.count+self.bs).astype(int) % len(self.images)
        self.imgs = self.images[indices]

        self.meta = self.sources[indices.astype(int)]
        self.count += self.bs

   
    def _check_alive(self)->None:
        """
        Check if threads are alive and re-open unresponsive streams.
        
        Note:
        -----
        Checks if threads are alive and re-opens unresponsive streams.
        """
        if self.count == len(self.sources):
            raise StopIteration
if __name__=="__main__":
    from data.loaders import DataLoaderImages

    # Load a dataset
    ds = DataLoaderImages("captured_images/")