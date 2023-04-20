import pandas as pd
import h5py
import numpy as np


def save_dataframe_to_h5_file(filename: str, dataframe: pd.DataFrame, name: str = None):
    with h5py.File(filename, "a") as hf:
        data = dataframe.to_numpy(dtype="float64")
        index = dataframe.index.to_numpy(dtype="S")
        columns = dataframe.columns.to_numpy(dtype="S")
        hf.create_dataset("data", data=data, dtype=data.dtype)
        hf.create_dataset("index", data=index, dtype=index.dtype)
        hf.create_dataset("columns", data=columns, dtype=columns.dtype)
        if name is not None:
            hf.create_dataset(
                "name", data=np.array([name]), dtype="<U{}".format(len(name))
            )
        hf.close()
