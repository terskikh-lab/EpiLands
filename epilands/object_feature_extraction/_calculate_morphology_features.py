import numpy as np
import numpy as np
from scipy.ndimage import center_of_mass
from skimage.measure import moments


def roundness(mask):
    """
    Calculates the roundness of a binary mask of a numpy array.

    Parameters:
    -----------
    mask : numpy.ndarray
        A binary mask of a NumPy array.

    Returns:
    --------
    float
        The roundness of the binary mask.
    """

    # Calculate the perimeter of the mask
    perimeter = np.sum(mask[:, 1:] != mask[:, :-1]) + np.sum(
        mask[1:, :] != mask[:-1, :]
    )

    # Calculate the area of the mask
    area = np.sum(mask)

    # Calculate the roundness of the mask
    roundness = (4 * np.pi * area) / (perimeter**2)

    return roundness


def eccentricity(mask):
    """
    Calculates the eccentricity of a binary mask of a numpy array.

    Parameters:
    -----------
    mask : numpy.ndarray
        A binary mask of a NumPy array.

    Returns:
    --------
    float
        The eccentricity of the binary mask.
    """

    # Calculate the moments of the mask
    m = moments(mask)

    # Calculate the centroid of the mask
    cy, cx = center_of_mass(mask)

    # Calculate the eccentricity of the mask
    eccentricity = (
        np.sqrt(1 - ((m[2, 0] - m[0, 2]) ** 2 / ((m[2, 0] + m[0, 2]) ** 2)))
        * (m[0, 0] / m[1, 1])
        * np.sqrt(
            (m[2, 0] + m[0, 2] + np.sqrt((m[2, 0] - m[0, 2]) ** 2 + 4 * m[1, 1] ** 2))
            / (m[2, 0] + m[0, 2] - np.sqrt((m[2, 0] - m[0, 2]) ** 2 + 4 * m[1, 1] ** 2))
        )
    )

    return eccentricity
