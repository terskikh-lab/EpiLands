import basicpy
import numpy as np


def fit_illumination_correction_model(group_images: np.array):
    """
    Fits an illumination correction model to a group of images using the BaSiC algorithm.

    Args:
        group_images (np.array): A numpy array containing the group of images to fit the model to.

    Returns:
        basicpy.BaSiC: The fitted illumination correction model.
    """
    correction_model = basicpy.BaSiC()
    correction_model.get_darkfield = True
    correction_model.fit(group_images)
    return correction_model
