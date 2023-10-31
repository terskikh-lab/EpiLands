import numpy as np
from PIL import Image
from typing import Union


def resize_image(image: np.ndarray, resize_factor: Union[int, float]) -> np.ndarray:
    """
    Resizes an image by a given factor.

    Args:
        image (np.ndarray): The image to resize.
        resize_factor (Union[int, float]): The factor by which to resize the image.

    Returns:
        np.ndarray: The resized image.
    """
    return np.array(
        Image.fromarray(image).resize((int(i * resize_factor) for i in image.shape))
    )
