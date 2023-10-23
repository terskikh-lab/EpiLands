import os
import pandas as pd
import numpy as np
from epilands import feature_model_construction
from ..tags import receives_data, outputs_data, receives_distance_matrix
from ...generic_tags import recieves_variable_attribute


@outputs_data
@receives_distance_matrix
@receives_data
def MIEL_distance(
    data: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    group_by=["ExperimentalCondition"],
    reference_group_A=(
        "Young"
    ),  # This group will be on the left, usually the young group
    reference_group_B=("Old"),  # This group will be on the right, usually the old group
):
    df_MIEL_distances = feature_model_construction.generate_MIEL_distances(
        df_pdist=distance_matrix,
        reference_group_A=reference_group_A,
        reference_group_B=reference_group_B,
        group_col_list=group_by,
    )
    return pd.merge(data, df_MIEL_distances, on=df_MIEL_distances.index.levels)
