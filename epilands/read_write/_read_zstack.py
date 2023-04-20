import os
import numpy as np
import pandas as pd
import logging
import tifffile as tiff
from typing import Union, Dict, AnyStr
from concurrent.futures import ThreadPoolExecutor
from ..image_preprocessing import calculate_2D_projection

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def read_zstack(
    image_files: Union[list, np.ndarray, pd.Series], return_3d: bool = False
):
    if not isinstance(image_files, pd.Series):
        image_files = pd.Series(image_files)
    image_files = image_files.sort_values()
    if len(image_files) == 1:
        tmpImg = tiff.imread(files=image_files.values[0])
    elif len(image_files) > 1 and return_3d == False:
        tmpImg = calculate_2D_projection(files=image_files)
    elif len(image_files) > 1 and return_3d == True:
        full_stack = [tiff.imread(file) for file in image_files]
        tmpImg = np.empty(shape=(len(image_files), *full_stack[0].shape))
        for i, img in enumerate(full_stack):
            tmpImg[i] = img
    else:
        raise
    return tmpImg


def load_images(
    file_information: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    with ThreadPoolExecutor() as executor:
        image_data = {
            channel: executor.submit(read_zstack, data["file_path"])
            for channel, data in file_information.groupby(["channel"])
        }
    for image in image_data:
        image_data[image] = image_data[image].result()
    return image_data
