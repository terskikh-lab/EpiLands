import os
import pandas as pd
import numpy as np
from epilands import tools
from ..tags import (
    receives_data,
    outputs_data,
)
from ...generic_tags import recieves_variable_attribute, outputs_variable_attribute
from typing import List, Callable
import logging

logger = logging.getLogger(name=__name__)


@outputs_data
@receives_data
def aggregate_data(
    data: pd.DataFrame,
    group_by: List[str],
    metric: Callable = np.mean,
):
    if not callable(metric):
        raise ValueError("metric must be a callable function")

    data_num, data_cat = tools.split_df_by_dtype(df=data)
    data_cat_new = tools.aggregate_df(data_cat, group_by=group_by, func="unique")
    data_num_new = tools.aggregate_df(data_num, group_by=group_by, func=metric)
    # for col in ["WellIndex", "Row", "Column", "FieldOfView", "XCoord", "YCoord"]:
    #     try:
    #         observations_num_new.drop(
    #             columns=col,
    #             inplace=True,
    #         )
    #     except KeyError:
    #         print(f"column {col} doesnt exist")
    # for col in observations_num_new.columns:
    #     if col in observations_cat_new.columns:
    #         observations_cat_new.drop(columns=col, inplace=True)
    data_new = data_cat_new.merge(
        data_num_new,
        left_index=True,
        right_index=True,
        validate="1:1",
    )
    data_new.reset_index(drop=False, inplace=True)
    # if isinstance(group_by, list) and len(group_by) == 1:  # WE NEED A BETTER S
    #     group_by = group_by[0]
    # feature_cols = data.columns
    # data[group_by] = observations[group_by]
    # data_new = data.groupby(group_by).agg(metric).dropna(axis=0, how="all")
    # data_new["Cell_Count"] = data.loc[:, group_by].value_counts(
    #     sort=False
    # )  # add cellcount
    # df_collapsed = data_new.merge(
    #     observations_new.set_index(group_by),
    #     left_index=True,
    #     right_index=True,
    #     validate="1:1",
    # )
    # df_collapsed.reset_index(inplace=True, drop=False)
    # feature_cols = feature_cols
    # observation_cols = df_collapsed.columns[~df_collapsed.columns.isin(feature_cols)]
    logger.info(f"Collapsed by:{group_by}\nmetric:{metric.__name__}")
    return data_new
