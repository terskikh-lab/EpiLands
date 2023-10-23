import os
import re
from typing import Union
import pandas as pd
import numpy as np
from epilands import tools, feature_preprocessing
from ..tags import receives_data, outputs_data, outputs_distance_matrix
from ...generic_tags import recieves_variable_attribute


@outputs_distance_matrix
@receives_data
def distance_matrix(
    data: pd.DataFrame,
    subset: Union[list, np.ndarray, str, re.Pattern],
    regex: bool = True,
):
    if isinstance(subset, (str, re.Pattern)):
        subset = tools.get_columns(data, pattern=subset, regex=regex)
    data_pdist = feature_preprocessing.distance_matrix_pdist(data=data[subset])
    return data_pdist
