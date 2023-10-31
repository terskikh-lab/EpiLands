import numpy as np
from typing import Tuple
from ._cellselecter import cellselecter, cellselecter_ND


def extract_object(
    img: np.ndarray,
    mask: np.ndarray,
    objectIdx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a single object from an image and its corresponding mask.

    Args:
        img (np.ndarray): The input image.
        mask (np.ndarray): The input mask.
        objectIdx (int): The index of the object to extract.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the segmented object image and its corresponding binary label.
    """
    # get the image with 0 pixel margin
    object_img, object_label = cellselecter_ND(
        img=img, label=mask, margin=1, cellIdx=objectIdx
    )
    # set pixels outside of mask = 0
    seg_object_img = np.where(object_label == 1, object_img, 0)
    # set object label to binary
    object_label = object_label.astype(bool)
    return (seg_object_img, object_label)
