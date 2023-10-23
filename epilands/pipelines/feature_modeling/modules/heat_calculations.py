import os
import re
import pandas as pd
import numpy as np
from epilands import (
    tools,
    generic_read_write,
    feature_model_construction,
    feature_preprocessing,
)
from sklearn.preprocessing import MinMaxScaler


def heat(df_cv, epiage):
    df_cv_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(df_cv), columns=df_cv.columns, index=df_cv.index
    )
    corr = df_cv_norm.corrwith(epiage)
    corrnorm = corr / np.linalg.norm(corr.values)
    df_heat = df_cv_norm * corrnorm
    return df_heat
