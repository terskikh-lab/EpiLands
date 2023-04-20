import numpy as np
import pandas as pd
from numba import njit


def CV(x, axis=None):
    """Calculates the Coefficient of Variance (std/mean) of the given array using numpy"""
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    mean = np.where(mean == 0, np.finfo(type(mean)).resolution, mean)
    return std / mean


def CV_jit(x):
    """Calculates the Coefficient of Variance (std/mean) of the given array using numpy"""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        try:
            x = x.to_numpy(dtype=np.float64)
        except:
            print(f"Couldn't convert {x.dtype} to numpy, returning nan")
            return np.nan
    return _cv_jit(x)


CV_jit.__name__ == "CV"


@njit()
def _cv_jit(x):
    mean = x.mean()
    std = x.std()
    mean = np.where(mean == 0, np.finfo(type(mean)).resolution, mean)
    return np.divide(std, mean)
