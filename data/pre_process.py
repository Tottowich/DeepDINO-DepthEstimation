import cv2
import numpy as np

class LetterBox:
    """
    This class is responsible for resizing images and adding padding for tasks like detection, 
    instance segmentation, and pose recognition.
    
    Attributes:
    -----------
    new_shape : tuple, default=(640, 640)
        The shape to resize the image to.
    auto : bool, default=False
        Automatically calculate padding based on stride.
    scaleFill : bool, default=False
        Stretch the image to fill the shape.
    scaleup : bool, default=True
        Allow scaling up of the image.
    stride : int, default=32
        The stride for padding calculation if 'auto' is True.
    """
    
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        """
        Initialize the LetterBox object with the given parameters.
        
        Parameters:
        -----------
        new_shape : tuple, default=(640, 640)
            The new shape to resize the image to.
        auto : bool, default=False
            If True, automatically calculates padding based on stride.
        scaleFill : bool, default=False
            If True, stretches the image to fill the shape.
        scaleup : bool, default=True
            If True, allows scaling up of the image.
        stride : int, default=32
            The stride for padding calculation if 'auto' is set to True.
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        """
        Resize and add padding to the image, and optionally update labels.
        
        Parameters:
        -----------
        labels : dict, optional
            The labels associated with the image.
        image : array_like, optional
            The image to be resized and padded.
            
        Returns:
        --------
        array_like or dict
            The resized and padded image, or updated labels if labels were provided.
        """
        labels = labels or {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape if labels.get('rect_shape') is None else labels.pop('rect_shape')
        
        # Compute optimal dimensions and scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        r = min(r, 1.0) if not self.scaleup else r
        new_unpad = tuple(map(int, map(round, (shape[1] * r, shape[0] * r))))
        
        # Calculate padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = dw % self.stride, dh % self.stride
        elif self.scaleFill:
            dw, dh = 0, 0
            new_unpad = new_shape

        # Update padding and image
        dw, dh = dw // 2, dh // 2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Update labels if present
        if labels:
            labels = self._update_labels(labels, (r, r), dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        
        return img

    def _update_labels(self, labels, ratio, padw, padh):
        """
        Update the labels based on the new dimensions and padding.
        
        Parameters:
        -----------
        labels : dict
            The labels to be updated.
        ratio : tuple
            The width and height ratios for resizing.
        padw : float
            The width padding.
        padh : float
            The height padding.
        
        Returns:
        --------
        dict
            The updated labels.
        """
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels

# Converted to Totto-style and optimized for speed
