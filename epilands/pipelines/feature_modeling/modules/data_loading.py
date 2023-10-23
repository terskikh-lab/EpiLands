import re
import os
import numpy as np
import pandas as pd

# from config import (
#     loadPathRoot,
#     savePathRoot,
#     size_thresh_low,
#     size_thresh_high,
#     label_channels,
# )
from ..processes.IO import load_platemap, load_h5_feature_data, merge_platemap_with_data
from ..processes.bootstrap_data import bootstrap_data
from ...create_script_output_dirs import create_script_output_dirs
from epilands import feature_visualization, feature_preprocessing, generic_read_write
from sklearn.mixture import GaussianMixture


def MIEL72_load_thresholdCD3_bootstrap(
    loadPathRoot: str,
    savePathRoot: str,
    platemap_path: str,
    sheet_name: str,
    num_bootstraps=5,
    num_cells=200,
    seed=1,
):
    outputDir = create_script_output_dirs(
        savePathRoot=savePathRoot, script_name="thresholdCD3_bootstrap"
    )
    data = load_h5_feature_data(
        feature_extraction_directory=os.path.join(
            loadPathRoot, "feature_extraction"
        ),  # feature_extraction_directory="/Volumes/1TBAlexey2/test_results/test_2d_pipeline/feature_extraction",
        search_pattern=".hdf5",
    )

    platemap = load_platemap(
        platemap_path=platemap_path,
        sheet_name=sheet_name,
    )

    data = merge_platemap_with_data(data=data, platemap=platemap, return_dask=False)

    thresholds = feature_preprocessing.multiotsu(
        data=data["CD3_object_average_intensity"], classes=2
    )

    data["CD3_positive"] = data["CD3_object_average_intensity"] > thresholds[0]

    data_bootmean = bootstrap_data(
        data=data,
        subset=re.compile("TXT"),
        group_by=["Sample", "Age", "CD3_positive"],
        metric=np.mean,
        num_bootstraps=num_bootstraps,
        num_cells=num_cells,
        with_replacement=False,
        seed=seed,
    )

    data_bootstd = bootstrap_data(
        data=data,
        subset=re.compile("TXT"),
        group_by=["Sample", "Age", "CD3_positive"],
        metric=np.std,
        num_bootstraps=num_bootstraps,
        num_cells=num_cells,
        with_replacement=False,
        seed=seed,
    )

    data_bootcv = bootstrap_data(
        data=data,
        subset=re.compile("TXT"),
        group_by=["Sample", "Age", "CD3_positive"],
        metric=lambda x: np.std(x) / np.mean(x),
        num_bootstraps=num_bootstraps,
        num_cells=num_cells,
        with_replacement=False,
        seed=seed,
    )

    generic_read_write.save_dataframe_to_csv(
        data_bootmean, path=outputDir, filename="bootstrap_mean.csv"
    )
    generic_read_write.save_dataframe_to_csv(
        data_bootstd, path=outputDir, filename="bootstrap_std.csv"
    )
    generic_read_write.save_dataframe_to_csv(
        data_bootcv, path=outputDir, filename="bootstrap_cv.csv"
    )

    return outputDir


def LH11_load_thresholdLEF1_bootstrap(
    loadPathRoot: str,
    savePathRoot: str,
    platemap_path: str,
    sheet_name: str,
    num_bootstraps,
    num_cells,
    frac,
    with_replacement,
    seed,
):
    params = {
        "num_bootstraps": num_bootstraps,
        "num_cells": num_cells,
        "with_replacement": with_replacement,
        "frac": frac,
        "seed": seed,
    }
    outputDir = create_script_output_dirs(
        savePathRoot=savePathRoot, script_name="thresholdLEF1_bootstrap"
    )
    data = load_h5_feature_data(
        feature_extraction_directory=os.path.join(
            loadPathRoot, "feature_extraction"
        ),  # feature_extraction_directory="/Volumes/1TBAlexey2/test_results/test_2d_pipeline/feature_extraction",
        search_pattern=".hdf5",
    )

    platemap = load_platemap(
        platemap_path=platemap_path,
        sheet_name=sheet_name,
    )

    data = merge_platemap_with_data(data=data, platemap=platemap, return_dask=False)

    feature_visualization.threshold_objects_histogram(
        filename="size_thresholding_histogram",
        series=data["MOR_object_pixel_count"],
        # high_threshold=thresholds[0],
        high_threshold=np.inf,
        low_threshold=750,
        range=(0, 10000),
        title="size threshold",
        save_info=True,
        graph_output_folder=outputDir,
        cells_per_bin=200,
    )

    inliers = feature_preprocessing.size_threshold_df(
        df=data,
        high_threshold=np.inf,
        low_threshold=750,
        threshold_col="MOR_object_pixel_count",
        threshold_metric="values",
    )

    params["low_size_threshold"] = 750
    # if np.count_nonzero(inliers.values) < (num_cells):
    #     raise ValueError(
    #         f"Too few cells after thresholding: \n{inliers.value_counts()}"
    #     )
    data = data[inliers]
    # inliers = data.loc[
    #     (
    #         data["LEF-1_object_average_intensity"]
    #         < data["LEF-1_object_average_intensity"].quantile(0.99)
    #     ),
    #     "LEF-1_object_average_intensity",
    # ]

    # LEF1_gm = GaussianMixture(n_components=2).fit_predict(inliers.values.reshape(-1, 1))

    # thresholds = feature_preprocessing.multiotsu(data=inliers, classes=2)
    # feature_visualization.threshold_objects_histogram(
    #     filename="LEF1_thresholding_histogram",
    #     series=data["LEF-1_object_average_intensity"],
    #     # high_threshold=thresholds[0],
    #     high_threshold=250,
    #     low_threshold=0,
    #     range=(0, 1000),
    #     title="LEF1 threshold",
    #     save_info=True,
    #     graph_output_folder=outputDir,
    #     cells_per_bin=200,
    # )

    # params["lEF1_threshold"] = 250
    # data["LEF1_positive"] = data["LEF-1_object_average_intensity"] > 250

    txt_feats = data.columns[data.columns.str.contains("TXT")]
    txt_feats = txt_feats[~txt_feats.str.contains("LEF-1")]
    data.reset_index(drop=True, inplace=True)

    # Tukey Fences
    grouped_data = data.groupby(["Sample", "Age", "Replicate", "FieldOfView"])
    grouped_avecv = (
        grouped_data[txt_feats].apply(lambda df: df.std() / df.mean()).mean(axis=1)
    )
    tukey_data = []
    gated_data = []
    outliers = []
    for name, dat in grouped_avecv.groupby("Sample"):
        Q1 = dat.quantile(0.25)
        Q3 = dat.quantile(0.75)
        IQRfence = Q3 - Q1 * 1.5
        inliers = (dat < (Q1 - IQRfence)) & (dat > (Q3 + IQRfence))
        tukey_data.append(inliers)
        gated_data.append(dat[inliers])
        for i in inliers.index[~inliers].values:
            outliers.append(pd.Series(grouped_data.get_group(i).index))
    tukey_data = pd.concat(tukey_data)
    gated_data = pd.concat(gated_data)
    # grouped_data = data.groupby(["Sample", "Age", "Replicate"])
    # Minimum cellcount per datapoint
    grouped_data = data.drop(pd.concat(outliers).values).groupby(
        ["Sample", "Age", "Replicate"]
    )
    group_sizes = grouped_data.size()
    for i in group_sizes.index[~(group_sizes < 100)].values:
        outliers.append(pd.Series(grouped_data.get_group(i).index))

    # Final results
    grouped_data = data.drop(pd.concat(outliers).values).groupby(
        ["Sample", "Age", "Replicate"]
    )
    group_sizes = grouped_data.size()
    grouped_means = grouped_data[txt_feats].mean()
    grouped_cv = grouped_data[txt_feats].apply(lambda df: df.std() / df.mean())
    # data_bootmean, group_sizes = bootstrap_data(
    #     data=data,
    #     subset=re.compile("TXT"),
    #     group_by=["Sample", "Age", "LEF1_positive"],
    #     metric=np.mean,
    #     num_bootstraps=num_bootstraps,
    #     num_cells=num_cells,
    #     with_replacement=with_replacement,
    #     frac=frac,
    #     seed=seed,
    # )

    # data_bootstd = bootstrap_data(
    #     data=data,
    #     subset=re.compile("TXT"),
    #     group_by=["Sample", "Age", "LEF1_positive"],
    #     metric=np.std,
    #     num_bootstraps=num_bootstraps,
    #     num_cells=num_cells,
    #     with_replacement=False,
    #     seed=seed,
    # )

    # data_bootcv = bootstrap_data(
    #     data=data,
    #     subset=re.compile("TXT"),
    #     group_by=["Sample", "Age", "LEF1_positive"],
    #     metric=lambda x: np.std(x) / np.mean(x),
    #     num_bootstraps=num_bootstraps,
    #     num_cells=num_cells,
    #     with_replacement=False,
    #     seed=seed,
    # )
    pd.Series(params).to_csv(os.path.join(outputDir, "params.csv"))
    # pd.Series(group_sizes).to_csv(os.path.join(outputDir, "bootmean_group_sizes.csv"))
    # generic_read_write.save_dataframe_to_csv(
    #     data_bootmean, path=outputDir, filename="bootstrap_mean.csv"
    # )
    pd.Series(group_sizes).to_csv(os.path.join(outputDir, "well_group_sizes.csv"))
    generic_read_write.save_dataframe_to_csv(
        grouped_means, path=outputDir, filename="well_mean.csv"
    )
    generic_read_write.save_dataframe_to_csv(
        grouped_cv, path=outputDir, filename="well_cv.csv"
    )

    # generic_read_write.save_dataframe_to_csv(
    #     data_bootstd, path=outputDir, filename="bootstrap_std.csv"
    # )
    # generic_read_write.save_dataframe_to_csv(
    #     data_bootcv, path=outputDir, filename="bootstrap_cv.csv"
    # )

    return outputDir
