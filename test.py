from typing import List
import numpy as np
import cv2
import torch
import torchvision.utils as tv_utils

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
    grid_tensor = tv_utils.make_grid(tensor=tensor_images, nrow=nrows, padding=0, normalize=True, scale_each=True)
    
    # Convert the grid tensor to a numpy array
    grid_numpy = grid_tensor.permute(1, 2, 0).numpy()

    return grid_numpy

# Example usage
if __name__ == "__main__":
    # Generate some example images (replace these with your actual images)
    img1 = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
    img3 = np.random.randint(0, 255, (120, 180, 3), dtype=np.uint8)


    images = [img1, img2, img3]
    nrows = 2

    grid = grid_images(images, nrows)
    cv2.imshow('Image Grid', grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
