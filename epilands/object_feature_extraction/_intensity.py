import pandas as pd
import numpy as np
from typing import Dict
from beartype import beartype


@beartype
def calculate_intensity_features(image_data: Dict[str, np.ndarray]) -> pd.Series:
    object_intensity_features = {
        f"{ch}_object_average_intensity": calculate_object_ave_int(
            img=obj_img, mask=image_data["masks"]
        )
        for ch, obj_img in image_data.items()
        if ch != "masks"
    }
    return pd.Series(object_intensity_features)


import numpy as np
from numba import njit


def calculate_object_ave_int(img: np.ndarray, mask: np.ndarray) -> float:
    try:
        return np.mean(img, where=mask > 0)
    except:
        return np.NAN


# @njit
# def calculate_object_ave_int(img: np.ndarray) -> float:
#     try:
#         obj_bounds = np.where(img > 0)
#         obj = img[obj_bounds[0], :]
#         obj = obj[:, obj_bounds[1]]
#         return np.mean(obj)
#     except:
#         return np.NAN


# @njit
# def calculate_object_ave_int(
#     img: np.ndarray,
#     masks: np.ndarray,
#     objectIdx: int) -> float:
#     obj_bounds = np.where(masks==objectIdx)
#     obj = img[obj_bounds[0], obj_bounds[1]]
#     return np.mean(obj)
