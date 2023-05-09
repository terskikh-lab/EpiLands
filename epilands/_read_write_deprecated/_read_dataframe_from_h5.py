from __future__ import annotations
import os
import pandas as pd
import h5py
import warnings
from typing import Union

# RELATIVE IMPORTS #


def read_dataframe_from_h5(filename: Union[str, bytes, os.PathLike]):
    """
    Reads a .h5 file given from ELTA extraction into a pandas DataFrame.txt
    \nRequires that the .h5 file has keys 'data', 'index', 'columns'
    \nif key 'name' exists an attribute '.attrs['name']' will be added to the DataFrame
    """
    with h5py.File(filename, "r") as hf:
        data = hf["data"][:]
        index = hf["index"][:]
        columns = hf["columns"][:].astype(str)
        df = pd.DataFrame(data, index=index, columns=columns)
        if len(index) == 0:
            warnings.warn(
                "read_dataframe_from_h5: no data was found in file {}".format(filename)
            )
            return df
        if "name" in hf.keys():
            df.attrs["name"] = str(hf["name"][:][0])
        hf.close()
    df.loc[:, "Path"] = filename
    return df
