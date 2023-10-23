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


def cumulative_distance_per_group(
    data,
):
    # zscore = zscore_data(
    #     data=data,
    #     group_by=["Tissue"],
    #     subset="TXT",
    #     regex=True,
    # )
    # data.loc[:, zscore.columns] = zscore
    # zscore = None

    # data.loc[:, "timepoint"] = data["Sample"].str.extract("-([0-9]+)")
    # data.loc[:, "Sample"] = data["Sample"].str.replace("-[0-9]+", "", regex=True)
    relevant_cols = tools.get_columns(data, pattern="TXT", regex=True)
    sample_sums = []
    for name, df_grp in data.groupby(["Sample", "Age", "LEF1_positive"]):
        cumulative_sum = {}
        for ch in np.unique(relevant_cols.str.extract("([A-Za-z0-9-]+)_TXT").values):
            if ch == "LEF-1":
                ch = "AllChannels"
                dm = distance_matrix(
                    data=df_grp,
                    subset="TXT",
                    regex=True,
                )
            else:
                dm = distance_matrix(
                    data=df_grp,
                    subset=ch,
                    regex=True,
                )
            cumulative_sum[ch] = []
            idx = pd.Index(df_grp["timepoint"])
            dm = pd.DataFrame(data=dm, index=idx, columns=idx)

            timepoint_means = dm.groupby(["timepoint"]).mean().sort_index()
            grp_keys = list(timepoint_means.index)
            for i, grp in enumerate(grp_keys):
                if i == 0:
                    last_dist_to_self = timepoint_means.loc[grp, grp].mean()
                    cum_dist_traveled = 0
                    continue
                timepoint_dist = timepoint_means.loc[grp, grp_keys[i - 1]].mean()
                dist_traveled = timepoint_dist - last_dist_to_self
                cum_dist_traveled += dist_traveled
                cumulative_sum[ch].append(cum_dist_traveled)
                last_dist_to_self = timepoint_means.loc[grp, grp].mean()
                print(cum_dist_traveled)
        sample_cumsum = pd.DataFrame.from_dict(cumulative_sum)
        sample_cumsum[["Sample", "Age", "LEF1_positive"]] = name
        sample_sums.append(sample_cumsum)
    df_cumsum = pd.concat(sample_sums).reset_index()


import matplotlib.pyplot as plt
import seaborn as sns


# def plot_cumdist():
#     fig, ax = plt.subplots()
#     sns.stripplot(
#         x="Age",
#         y=0,
#         hue="Sample",
#         data=df_cumsum,
#     )
#     plt.legend()
#     sns.lineplot(
#         x="Age",
#         y=0,
#         hue="Sample",
#         data=df_cumsum,
#         ax=ax,
#         # order=['463 days', '491 days', '521 days', '544 days', '592 days',
#         #    '653 days', '681 days', '711 days', '734 days', '782 days']
#     )
#     plt.tight_layout()
#     plt.ylabel("Cumulative Distance")
#     ELTA.read_write.save_matplotlib_figure(
#         fig=fig,
#         figurename="all_mouse_cumulative_distance_lineplot_outlier_removed",
#         path=sciadata.uns["experiment_output_folder"],
#     )
#     df_cumsum.to_csv(
#         os.path.join(
#             sciadata.uns["experiment_output_folder"],
#             "all_mouse_cumulative_distance_lineplot_outlier_removed.csv",
#         )
#     )
