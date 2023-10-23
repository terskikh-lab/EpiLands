import os
import pandas as pd
import numpy as np
from epilands import feature_preprocessing
from ..tags import (
    receives_data,
    outputs_data,
)
from ...generic_tags import recieves_variable_attribute


@outputs_data
@receives_data
def threshold_objects(
    data: pd.DataFrame,
    high_threshold,
    low_threshold,
    threshold_col,
    threshold_metric,
):
    inlier_objects = feature_preprocessing.size_threshold_df(
        df=data,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        threshold_col=threshold_col,
        threshold_metric=threshold_metric,
    )
    data[f"{threshold_col}_{threshold_metric}_inliers"] = inlier_objects

    return data
