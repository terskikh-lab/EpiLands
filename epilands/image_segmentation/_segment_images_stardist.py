from stardist.models.model2d import StarDist2D
from stardist.models.model3d import StarDist3D
from typing import Tuple, Any
import numpy as np

stardist_model_2d = StarDist2D.from_pretrained("2D_versatile_fluo")
stardist_model_3d = StarDist3D.from_pretrained("3D_demo")


def segment_image_stardist2d(image: np.ndarray) -> Tuple[np.ndarray, list, dict]:
    """
    Segment a 2D image using a StarDist model.

    Args:
        image (np.ndarray): The input image to segment.

    Returns:
        Tuple[np.ndarray, list, dict]: A tuple containing the segmented masks, a list of objects, and segmentation details.
    """
    masks, details = stardist_model_2d.predict_instances(image)
    # count the unique masks and return objects and mask sizes
    objects, counts = np.unique(masks.reshape(-1, 1), return_counts=True, axis=0)
    # delete 0 as that labels background
    objects = list(np.delete(objects, 0))
    return masks, objects, details


def segment_image_stardist3d(image: np.ndarray) -> Tuple[np.ndarray, list, dict]:
    """
    Segment a 3D image using a Stardist model.

    Args:
        image (np.ndarray): A 3D numpy array representing the image to be segmented.

    Returns:
        Tuple[np.ndarray, list, dict]: A tuple containing the segmented masks as a numpy array, a list of objects, and a dictionary of details.
    """
    masks, details = stardist_model_3d.predict_instances(image)
    # count the unique masks and return objects and mask sizes
    objects, counts = np.unique(masks.reshape(-1, 1), return_counts=True, axis=0)
    # delete 0 as that labels background
    objects = list(np.delete(objects, 0))
    return masks, objects, details
