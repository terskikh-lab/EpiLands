import pandas as pd
from typing import Union
import numpy as np

from ._join_columns import join_columns
from ..config import COLUMN_SEPARATOR, NAME_SEPARATOR_


def aggregate_df(df: pd.DataFrame, groupby, func: Union[callable, str]) -> pd.DataFrame:
    df = df.copy()
    # if isinstance(groupby, str) or (not isinstance(groupby, str) and len(groupby) == 1):
    #     if isinstance(groupby, str):
    #         newname = COLUMN_SEPARATOR.join([groupby, "groups"])
    #     else:
    #         newname = COLUMN_SEPARATOR.join([groupby[0], "groups"])
    #     df[newname] = df[groupby]
    #     df_groupby = df.groupby(newname, as_index=True)
    #     df_agg = df_groupby.agg(func)
    if func == "unique":
        newname = NAME_SEPARATOR_.join(groupby)
        newname = NAME_SEPARATOR_.join([newname, "groups"])
        df[newname] = join_columns(df, columns=groupby)
        df_groupby = df.groupby(newname, as_index=True)
        aggregate_results = [
            i.aggregate(np.unique).apply(lambda l: l[0] if len(l) == 1 else np.NaN)
            for _, i in df_groupby
        ]
        df_agg = pd.concat(aggregate_results, axis=1).T.set_index(newname)
        df_agg.dropna(axis=1, how="all", inplace=True)
    else:
        newname = COLUMN_SEPARATOR.join(groupby)
        newname = COLUMN_SEPARATOR.join([newname, "groups"])
        df[newname] = join_columns(df, columns=groupby)
        df_groupby = df.groupby(newname, as_index=True)
        df_agg = df_groupby.agg(func)
    return df_agg


def _aggregate_unique_cols(df: pd.DataFrame):
    result = {}
    for col in df.columns:
        unique_data = df[col].unique()
        if len(unique_data) == 1:
            result[col] = unique_data[0]
        else:
            result[col] = np.NaN
    return pd.Series(result)
