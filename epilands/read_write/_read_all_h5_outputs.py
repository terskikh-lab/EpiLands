from __future__ import annotations
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# RELATIVE IMPORTS #
from ._read_dataframe_from_h5_file import read_dataframe_from_h5_file
from ._find_all_files import find_all_files


def read_all_h5_outputs(
    feature_extraction_directory: str,
    raw_data_pattern: str = "_features",
) -> pd.DataFrame:
    """
    This function reads all files matching a particular pattern within a given directory.
    \nNote that currently the function includes all file types matching the search pattern.

    file_folder_loc: str, the location of the folder containing the files
    raw_data_pattern: str, the pattern to search for in the file names
    """
    files = find_all_files(feature_extraction_directory, raw_data_pattern)
    # start = time.perf_counter()
    # data = []
    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(read_dataframe_from_h5_file, f) for f in files]
    #     for filedata in tqdm(as_completed(futures)):
    #         data.append(filedata.result())
    #     executor.shutdown(wait=True)
    # stop = time.perf_counter()
    # print(stop - start)

    start = time.perf_counter()
    data = []
    for file in tqdm(files):
        filedata = read_dataframe_from_h5_file(file)
        if filedata.shape[0] == 0:
            continue
        else:
            data.append(filedata)
    stop = time.perf_counter()
    print(stop - start)
    data = pd.concat(data, ignore_index=True)
    return data
