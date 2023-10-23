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
import matplotlib.pyplot as plt
from ...create_script_output_dirs import create_script_output_dirs
from ..processes.IO import load_platemap, load_h5_feature_data, merge_platemap_with_data
from ..processes.zscore_features import zscore_data
from ..processes.distance_matrix import distance_matrix
from ..processes.bootstrap_data import bootstrap_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


fit, *_ = feature_embedding.calculate_component_analysis(
    fit_data=data[relevant_cols],
    feature_data=data[relevant_cols],
    analysis_type="PCA",
    n_components=2,
)
relevant_cols = tools.get_columns(data, pattern="TXT", regex=True)
