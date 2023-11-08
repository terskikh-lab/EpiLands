import os
import numpy as np
import pandas as pd
import logging
import tifffile as tiff
from typing import Union, Dict, AnyStr
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def read_images(
    image_files: Union[list, np.ndarray, pd.Series], return_3d: bool = False
):
    if not isinstance(image_files, pd.Series):
        image_files = pd.Series(image_files)
    image_files = image_files.sort_values(
        ascending=True
    )  ### Generalize regex for usage using plane -> int sorting
    tmpImg = tiff.imread(files=image_files.to_list())
    if len(image_files) > 1 and return_3d == False:
        tmpImg = tmpImg.max(axis=0)
    return tmpImg
