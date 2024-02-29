import os
import re
import logging
from typing import Union, Optional
import pandas as pd
from ._parse_imagelike_filename_metadata import parse_imagelike_filename_metadata
from ._find_all_files import find_all_files

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def extract_imagelike_file_information(
    file_path: str,
    search_pattern: Union[str, re.Pattern],
    metadata_pattern: Union[str, re.Pattern],
    channelIndex: int,
    rowIndex: int,
    colIndex: int,
    zIndex: int,
    FOVIndex: Union[tuple, int],
    tIndex: int,
) -> pd.DataFrame:
    """
    Description load_file_paths:
    loads files into memory, saves their paths to a dict

    INPUTS #=====================================

    image_directory: str = master folder containing raw images to be analyzed. This folder may contain subfolders,
    they will all be searched.

    OUTPUTS #=====================================
    images_file_information: pd.DataFrame = dataframe containing file paths and well indexing data

    channel_info: list = list of channel names
    #================================================

    Martin Alvarez-Kuglen @ Sanford Burnham Prebys Medical Discovery Institute: 2022/01/07
    """
    if not os.path.isdir(file_path):
        raise ValueError(f"file_path does not exist: {file_path}")
    file_list = find_all_files(
        file_path,
        pattern=search_pattern,
    )
    logger.info(f"Found {len(file_list)} {search_pattern} files in the given directory")
    if len(file_list) == 0:
        raise ValueError(f"No image files found in directory: {file_path}")
    ## We will start by generating new names for the files
    file_information = parse_imagelike_filename_metadata(
        files=pd.Series(file_list),
        pattern=metadata_pattern,
        channelIndex=channelIndex,
        rowIndex=rowIndex,
        colIndex=colIndex,
        zIndex=zIndex,
        FOVIndex=FOVIndex,
        tIndex=tIndex,
    )
    if channelIndex is not None:
        # check if any of the channels have more/less images
        if len(file_information["channel"].value_counts().unique()) != 1:
            channelcounts = file_information["channel"].value_counts()
            raise ValueError(
                f"ERROR: MISSING OR EXTRA IMAGES. \n\nNumber of images given for one channel does not match number of images given for another channel. See Below: \n{channelcounts}"
            )
        # create a list of all channels detected
        channels = list(file_information["channel"].unique())
        # check if channels result in a non-integer division (ie, missing images)
        if (len(file_list) / len(channels)) != (len(file_list) // len(channels)):
            raise ValueError(
                "ERROR: NUMBER OF IMAGES PROVIDED DOES NOT AGREE WITH NUMBER OF CHANNELS DETECTED. \
                PLEASE CHECK INPUT IMAGES"
            )
        logger.info(f"Channels Detected: {channels}")

    return file_information
