
from typing import Optional, List, Union
import cv2
import numpy as np
import torch
from torchvision import transforms as T
import torchvision as tv
import time

BASE_HEIGHT = 640
BASE_WIDTH = 480

class Visualizer:
    """
    A generic visualizer for images.
    
    Attributes:
    -----------
    image_name : str
        The title of the window in which the image will be displayed.
    rescale : bool
        Whether to rescale the image before displaying.
    _last_update_time : Optional[float]
    """
    
    def __init__(self, image_name: str = "Image", height:int=BASE_HEIGHT, width:int=BASE_WIDTH,rescale: bool = False):
        """
        Initialize the Visualizer object.
        
        Parameters:
        -----------
        image_name : str, default="Image"
            The title of the window in which the image will be displayed.
        rescale : bool, default=False
            Whether to rescale the image before displaying.
        """
        self.image_name = image_name
        self.rescale = rescale

        self._height = height
        self._width = width
        self._last_update_time = None

        self._initiate()

    def _initiate(self):
        """
        Initializes the window for displaying the image.
        """
        cv2.namedWindow(self.image_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.image_name, BASE_HEIGHT, BASE_WIDTH)
        self._last_update_time = time.monotonic()

    def update(self, img0: np.ndarray, **kwargs):
        """
        Update the image in the window.
        
        Parameters:
        -----------
        img0 : np.ndarray
            The image to be displayed.
        
        Returns:
        --------
        bool
            True if the window is still open, False otherwise.
        """
        if not self._check_window_closed():
            return False
        self.preprocess_image(img0.copy(), **kwargs)
        self.compute_fps(img0)
        self.display_image(img0)
        return True
    def _check_window_closed(self):
        """
        Checks if the window is closed or if 'q'/ESC was pressed.
        """
        if cv2.getWindowProperty(self.image_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
        elif cv2.waitKey(1) in {ord('q'), 27}:
            return False
        return True

    def preprocess_image(self, img0: np.ndarray, **kwargs):
        """
        Preprocess the image before displaying. Generally, this method should be overridden in subclasses.
        
        Parameters:
        -----------
        img0 : np.ndarray
            The image to be displayed.
        """
        return img0
    
    def compute_fps(self, img0):
        """
        Computes the FPS of the video stream and displays it on the image.
        """
        current_time = time.monotonic()

        if self._last_update_time is not None:
            fps = 1 / (current_time - self._last_update_time)
            cv2.putText(img0, f'FPS: {fps:.2f}', (img0.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self._last_update_time = current_time

    def display_image(self, img0: np.ndarray):
        """
        Display the image in a window.
        
        Parameters:
        -----------
        img0 : np.ndarray
            The image to be displayed.
        """
        if self.rescale:
            img0 = cv2.resize(img0, (self._height, int(self.width / img0.shape[1] * img0.shape[0])))
        cv2.imshow(self.image_name, img0)
        cv2.waitKey(1)

def grid_images(images: List[np.ndarray], nrows: int) -> np.ndarray:
    """
    Create a grid of images with nrows number of images per row.
    
    Parameters:
    -----------
    images : List[np.ndarray]
        List of images as numpy arrays.
    nrows : int
        Number of images per row in the grid.
    
    Returns:
    --------
    np.ndarray
        A numpy array representing the grid of images.
    """

    # Find the maximum dimensions among all images
    max_h = max([img.shape[0] for img in images])
    max_w = max([img.shape[1] for img in images])

    # Pad each image to the maximum dimensions
    padded_images = []
    for img in images:
        pad_h = max_h - img.shape[0]
        pad_w = max_w - img.shape[1]
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
        padded_images.append(padded_img)

    # Convert the list of numpy images to a tensor
    tensor_images = [torch.from_numpy(img).float().permute(2, 0, 1) for img in padded_images]
    tensor_images = torch.stack(tensor_images)
    # Use torchvision to make a grid
    grid_tensor = tv.utils.make_grid(tensor=tensor_images, nrow=nrows, padding=0, normalize=True, scale_each=True)
    
    # Convert the grid tensor to a numpy array
    grid_numpy = np.ascontiguousarray(grid_tensor.permute(1, 2, 0).numpy())

    return grid_numpy