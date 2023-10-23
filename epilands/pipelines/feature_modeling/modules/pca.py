import os
import pandas as pd
import epilands as el
import matplotlib.pyplot as plt
from workspace.feature_modeling.processes.zscore_features import zscore_data
from ...create_script_output_dirs import create_script_output_dirs

# data = pd.read_csv(
#     "/Volumes/MartinSSD2TB/MIEL72_recapitulation/MIEL72_PBMC_clock/module1_load_merge_bootstrap/bootstrap_cv.csv"
# )


def MEIL72_pca(
    data,
    group_by,
    seed,
    subset: str,
):
    # li = df_component_analysis["Age"].unique().tolist()
    # li.sort()
    # cmap = el.feature_visualization.create_color_mapping(
    #     legend_items=li,
    #     package="plt",
    #     map_type="continuous",
    # )

    # fig = el.feature_visualization.plot_component_pie_and_scatter(
    #     df_component_analysis=df_component_analysis,
    #     df_components=df_components,
    #     title="MEAN_PCA",
    #     figurename="mean_PCA",
    #     color_by="Age",
    #     color_map=cmap,
    #     channels=["CD3", "DAPI", "H3K4me1"],
    #     channel_color_map={"CD3": "r", "DAPI": "b", "H3K4me1": "g"},
    #     graph_output_folder=outputDir,
    #     save_info=True,
    #     shape_by="CD3_positive",
    #     size=80,
    # )
    return df_component_analysis, df_components
