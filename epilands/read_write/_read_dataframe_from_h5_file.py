import pandas as pd
import h5py


def read_dataframe_from_h5_file(filename: str) -> pd.DataFrame:
    with h5py.File(filename, "r") as hf:
        data = hf["data"][:]
        index = hf["index"][:]
        columns = hf["columns"][:].astype(str)
        df = pd.DataFrame(data, index=index, columns=columns)
        if "name" in hf.keys():
            df.attrs["name"] = str(hf["name"][:][0])
        hf.close()
    df.loc[:, "Path"] = filename
    return df
