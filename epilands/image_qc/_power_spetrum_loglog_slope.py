import logging
import numpy as np
from scipy.stats import linregress
from typing import Union
import os

from ._power_spectrum import power_spectrum

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)

## Image Quality Control


def power_spectrum_loglog_slope(
    image: np.ndarray, plot_result: bool = True
) -> Union[int, float]:
    """
    Calculates the slope of the power spectrum of an image using a log-log plot.

    Args:
        image (np.ndarray): The input image.
        plot_result (bool): Whether to plot the power spectrum and the linear regression line.

    Returns:
        The slope of the power spectrum.

    """
    logger.info("Calculating power spectrum")
    kvals, Abins = power_spectrum(image)
    slope, intercept, r_value, p_value, std_err = linregress(
        np.log10(kvals), np.log10(Abins)
    )
    logger.info(f"Power spectrum slope: {slope}")
    return slope
