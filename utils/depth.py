from typing import Union
import numpy as np
import torch
from matplotlib import colormaps


def render_depth(values: torch.Tensor, colormap_name: str = "magma_r") -> np.ndarray:
    """
    Renders a depth map using a given colormap.

    Takes a 2D array representing depth values and converts it to a 3D array representing a colored image,
    using a specified colormap.

    Parameters:
    -----------
    values : torch.Tensor
        A 2D numpy array of shape (H, W) containing depth values.
    colormap_name : str, default="magma_r"
        The name of the colormap to use. Must be one available in matplotlib.colormaps.

    Returns:
    --------
    np.ndarray
        A 3D numpy array of shape (H, W, 3) representing the colored depth map.

    Example:
    --------
    >>> depth_map = np.array([[0.1, 0.2], [0.2, 0.4]])
    >>> colored_map = render_depth(depth_map)
    """
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)  # Shape: (H, W, 4)
    colors = colors[:, :, :3]  # Discard alpha component, Shape: (H, W, 3)
    
    return colors
