from __future__ import annotations
import pandas as pd
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import time

# RELATIVE IMPORTS #
from ._read_dataframe_from_h5_file import read_dataframe_from_h5_file
from ._find_all_files import find_all_files


# def read_all_h5_outputs(
#     feature_extraction_directory: str,
#     platemap: pd.DataFrame,
#     raw_data_pattern: str = "_features",
# ) -> pd.DataFrame:
#     """
#     This function reads all files matching a particular pattern within a given directory.
#     \nNote that currently the function includes all file types matching the search pattern.

#     file_folder_loc: str, the location of the folder containing the files
#     raw_data_pattern: str, the pattern to search for in the file names
#     """
#     files = find_all_files(feature_extraction_directory, raw_data_pattern)
#     data = []
#     for file in tqdm(files):
#         filedata = read_dataframe_from_h5(file)
#         if filedata.shape[0] == 0:
#             continue
#         else:
#             data.append(filedata)
#     data = pd.concat(data, ignore_index=True)
#     check_shape = data.shape[0]
#     # data["WellIndex"] = data["WellIndex"]  # .map(lambda x: list(str(x))).map(lambda x: x[:1]+x[2:]).map(lambda x: ''.join(x)).astype(float).astype(int)
#     data = platemap.merge(data, how="inner", left_on="WellIndex", right_on="WellIndex")

#     if data.shape[0] != check_shape:
#         # return raw_data, platemap##FOR ERROR TESTING
#         raise ValueError(
#             "ERROR: SOME CELLS WERE LOST IN MERGE\n"
#             + f"(shape before:{check_shape} shape after:{data.shape[0]}).\n"
#             + "CHECK PLATEMAP AND RAW DATA INPUTS FOR MISSING ENTRIES"
#         )
#     return data


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
    #     futures = executor.map(read_dataframe_from_h5_file, files)
    #     for filedata in tqdm(futures):
    #         if filedata.shape[0] == 0:
    #             continue
    #         else:
    #             data.append(filedata)
    #     executor.shutdown(wait=True)
    # stop = time.perf_counter()
    # print(stop - start)

    # start = time.perf_counter()
    data = []
    for file in tqdm(files):
        filedata = read_dataframe_from_h5_file(file)
        if filedata.shape[0] == 0:
            continue
        else:
            data.append(filedata)
    # stop = time.perf_counter()
    # print(stop - start)
    data = pd.concat(data, ignore_index=True)
    return data
