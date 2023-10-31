import numpy as np
from typing import Union


# Description glaylconvert
# img ==> numpy array
# Kenta Ninomiya @ Kyushu University: 2021/3/19
def glaylconvert(
    img: np.ndarray,
    orgLow: Union[int, float],
    orgHigh: Union[int, float],
    qLow: Union[int, float],
    qHigh: Union[int, float],
) -> np.ndarray:
    """
    Convert a grayscale image to q-space.

    Parameters:
    img (np.ndarray): The grayscale image to be converted.
    orgLow (Union[int, float]): The original low grayscale level.
    orgHigh (Union[int, float]): The original high grayscale level.
    qLow (Union[int, float]): The low q-space level.
    qHigh (Union[int, float]): The high q-space level.

    Returns:
    np.ndarray: The converted q-space image.
    """
    # Quantization of the grayscale levels in the ROI
    img = np.where(img > orgHigh, orgHigh, img)
    img = np.where(img < orgLow, orgLow, img)
    cImg = ((img - orgLow) / (orgHigh - orgLow)) * (qHigh - qLow) + qLow
    return cImg
