import numpy as np
from typing import List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from ._extract_object import extract_object


def object_cookie_cutter(
    image: np.ndarray,
    mask: np.ndarray,
    objects: List[Union[int, float]],
) -> Tuple[dict, dict]:
    """
    Extracts objects from an image using a mask and returns a dictionary of object images and masks.

    Args:
        image (np.ndarray): The input image.
        mask (np.ndarray): The mask to apply to the image.
        objects (List[Union[int, float]]): A list of object IDs to extract from the image.

    Returns:
        Tuple[dict, dict]: A tuple containing two dictionaries. The first dictionary contains the extracted object images,
        keyed by object ID. The second dictionary contains the corresponding object masks, also keyed by object ID.
    """
    # start=time.perf_counter() # Use these for testing the speed of the function
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            extract_object,
            repeat(image),
            repeat(mask),
            objects,
        )
    object_images = {}
    object_masks = {}
    for objectIdx, result in zip(objects, results):
        object_images[objectIdx] = result[0]
        object_masks[objectIdx] = result[1]
    # finish = time.perf_counter()
    # print('Finished in {} seconds'.format(finish-start))
    # return finish-start
    return object_images, object_masks
