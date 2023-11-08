import logging
import numpy as np
import os

# RELATIVE IMPORTS #
from ..config import COLUMN_SEPARATOR

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def join_iterable(t):
    if isinstance(t, (list, tuple, np.ndarray)):
        if len(t) == 1:
            return str(t[0])
        elif len(t) > 1:
            t = [str(i) for i in t if i != None]
            return COLUMN_SEPARATOR.join(t)
        else:
            logger.error("{} was empty: {}".format(type(t), t))
    if isinstance(t, (str, type(None))):
        return t
    if isinstance(t, (float, int)):
        return str(t)
    else:
        raise ValueError(
            f"join_tuple only accepts types {[list, tuple, str]} but {type(t)} was given"
        )
