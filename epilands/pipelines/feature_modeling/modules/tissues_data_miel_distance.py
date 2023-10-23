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
import logging
from ...create_script_output_dirs import create_script_output_dirs
from ..processes.IO import load_platemap, load_h5_feature_data, merge_platemap_with_data
from ..processes.zscore_features import zscore_data
from ..processes.bootstrap_data import bootstrap_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("modules")


def chromage_with_accuracy(
    data,
    group_by,
    reference_group_A,
    reference_group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting chromage_with_accuracy using seed: {seed}")
    test_size = 0.25
    train_size = 0.75

    relevant_features = data.columns[data.columns.str.contains(subset)]
    data_grouped = data.groupby(["Sample", "ExperimentalCondition"])
    data = None
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    test_grouped = test_data.groupby(["Sample", "ExperimentalCondition"])

    data_groups = list(data_grouped.groups.keys())
    test_groups = list(test_grouped.groups.keys())
    if any([i not in data_groups for i in test_groups]):
        raise ValueError("Not enough cells to continue")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")

    train_data, train_group_sizes = bootstrap_data(
        data=train_data,
        subset=subset,
        group_by=["Sample", "ExperimentalCondition"],
        metric="mean",
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )

    test_data, test_group_sizes = bootstrap_data(
        data=test_data,
        subset=subset,
        group_by=["Sample", "ExperimentalCondition"],
        metric="mean",
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )
    group_sizes = []
    for n, d in zip(["train", "test"], [train_group_sizes, test_group_sizes]):
        d = d.to_frame().rename({0: "n"}, axis=1)
        d["split"] = n
        group_sizes.append(d)
    group_sizes = pd.concat(group_sizes, axis=0)

    for col, refA, refB in zip(group_by, reference_group_A, reference_group_B):
        test_data_middlage = test_data.loc[~test_data[col].isin([refA, refB]), :]
        train_data = train_data.loc[train_data[col].isin([refA, refB]), :]
        test_data = test_data.loc[test_data[col].isin([refA, refB]), :]

    df_groupby = train_data.groupby(group_by, as_index=True)[relevant_features]
    if len(group_by) == 1:
        reference_group_A = reference_group_A[0]
        reference_group_B = reference_group_B[0]
    A_centroid = df_groupby.get_group(reference_group_A).mean()
    A_centroid.attrs["name"] = f"{tools.join_iterable(reference_group_A)} centroid"
    B_centroid = df_groupby.get_group(reference_group_B).mean()
    B_centroid.attrs["name"] = f"{tools.join_iterable(reference_group_B)} centroid"

    epiAgeVec = B_centroid - A_centroid

    train_MIEL_distance = (
        feature_model_construction.generate_MIEL_distance_centroidvector(
            df=train_data[relevant_features],
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )

    if len(group_by) > 1:
        train_y_true = np.logical_and(
            train_data[col] == val for col, val in zip(group_by, reference_group_B)
        )
        test_y_true = np.logical_and(
            test_data[col] == val for col, val in zip(group_by, reference_group_B)
        )
    else:
        train_y_true = train_data[group_by] == reference_group_B
        test_y_true = test_data[group_by] == reference_group_B

    y_score = train_MIEL_distance["MIEL_distance"].values.reshape(-1, 1)

    fpr, tpr, thresholds = roc_curve(
        y_true=train_y_true,
        y_score=y_score,
    )

    auc = roc_auc_score(train_y_true, y_score)

    if auc < 1:
        threshold = thresholds[np.argmin((tpr - 1) ** 2 + fpr**2)]
    else:
        threshold = (y_score[train_y_true].min() + y_score[~train_y_true].max()) / 2

    train_y_pred = y_score > threshold

    train_accuracy = accuracy_score(train_y_true, train_y_pred)

    train_confusion = pd.Series(
        data=confusion_matrix(train_y_true, train_y_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    test_MIEL_distance = (
        feature_model_construction.generate_MIEL_distance_centroidvector(
            df=test_data[relevant_features],
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )

    y_score = test_MIEL_distance["MIEL_distance"].values.reshape(-1, 1)

    test_y_pred = y_score > threshold

    test_accuracy = accuracy_score(test_y_true, test_y_pred)

    test_confusion = pd.Series(
        data=confusion_matrix(test_y_true, test_y_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    result = pd.Series(
        {
            "auc": auc,
            "threshold": threshold,
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
        }
    )

    test_middleage_MIEL_distance = (
        feature_model_construction.generate_MIEL_distance_centroidvector(
            df=test_data_middlage[relevant_features],
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )

    train_MIEL_distance[["Sample", "Age", "ExperimentalCondition"]] = train_data[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    test_MIEL_distance[["Sample", "Age", "ExperimentalCondition"]] = test_data[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    test_middleage_MIEL_distance[
        ["Sample", "Age", "ExperimentalCondition"]
    ] = test_data_middlage[["Sample", "Age", "ExperimentalCondition"]]

    return (
        result,
        train_confusion,
        test_confusion,
        epiAgeVec,
        train_MIEL_distance,
        test_MIEL_distance,
        test_middleage_MIEL_distance,
        group_sizes,
    )


def heat(df_cv, chromage):
    df_cv_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(df_cv), columns=df_cv.columns, index=df_cv.index
    )
    corr = df_cv_norm.corrwith(chromage).fillna(0)
    corrnorm = corr / np.linalg.norm(corr.values)  # .abs()
    ser_heat = df_cv_norm.dot(corrnorm)
    ser_heat.name = "HEAT"
    return ser_heat.to_frame(), corrnorm


def chromageheat(df_cv, epiAgeAxis):
    df_cv_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(df_cv), columns=df_cv.columns, index=df_cv.index
    )
    epiAgeAxisNorm = (epiAgeAxis / np.linalg.norm(epiAgeAxis.values)).abs()
    ser_heat = df_cv_norm.dot(epiAgeAxisNorm)
    ser_heat.name = "HEAT"
    return ser_heat.to_frame()


def chromage_with_accuracy_and_heat(
    data,
    group_by,
    reference_group_A,
    reference_group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting chromage_with_accuracy using seed: {seed}")
    test_size = 0.25
    train_size = 0.75

    relevant_features = data.columns[data.columns.str.contains(subset)]

    zscore = zscore_data(data=data, group_by=None, subset=subset)
    new = data.columns[~data.columns.isin(zscore.columns)]
    zscore.loc[:, new] = data.loc[:, new]

    data_grouped = zscore.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    test_grouped = test_data.groupby(["Sample", "ExperimentalCondition"])

    data_groups = list(data_grouped.groups.keys())
    test_groups = list(test_grouped.groups.keys())
    if any([i not in data_groups for i in test_groups]):
        raise ValueError("Not enough cells to continue")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")

    def cv(col):
        m = col.mean()
        if m != 0:
            return col.std() / m
        else:
            return 0

    test_cv, test_cv_group_sizes = bootstrap_data(
        data=data.loc[test_data.index, :],
        subset=subset,
        group_by=["Sample", "ExperimentalCondition"],
        metric=cv,
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )
    test_cv.fillna(0, inplace=True)

    data = None
    zscore = None

    train_data, train_group_sizes = bootstrap_data(
        data=train_data,
        subset=subset,
        group_by=["Sample", "ExperimentalCondition"],
        metric="mean",
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )

    test_data, test_group_sizes = bootstrap_data(
        data=test_data,
        subset=subset,
        group_by=["Sample", "ExperimentalCondition"],
        metric="mean",
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )

    group_sizes = []
    for n, d in zip(["train", "test"], [train_group_sizes, test_group_sizes]):
        d = d.to_frame().rename({0: "n"}, axis=1)
        d["split"] = n
        group_sizes.append(d)
    group_sizes = pd.concat(group_sizes, axis=0)

    for col, refA, refB in zip(group_by, reference_group_A, reference_group_B):
        train_data = train_data.loc[train_data[col].isin([refA, refB]), :]
        test_data_middlage = test_data.loc[~test_data[col].isin([refA, refB]), :]
        # test_cv_middleage = test_cv.loc[~test_cv[col].isin([refA, refB]), :]
        test_data = test_data.loc[test_data[col].isin([refA, refB]), :]
        # test_cv = test_cv.loc[test_cv[col].isin([refA, refB]), :]
        test_cv_youngold = test_cv.loc[test_cv[col].isin([refA, refB]), :]

    df_groupby = train_data.groupby(group_by, as_index=True)[relevant_features]
    if len(group_by) == 1:
        reference_group_A = reference_group_A[0]
        reference_group_B = reference_group_B[0]
    A_centroid = df_groupby.get_group(reference_group_A).mean()
    A_centroid.attrs["name"] = f"{tools.join_iterable(reference_group_A)} centroid"
    B_centroid = df_groupby.get_group(reference_group_B).mean()
    B_centroid.attrs["name"] = f"{tools.join_iterable(reference_group_B)} centroid"

    epiAgeVec = B_centroid - A_centroid

    train_MIEL_distance = (
        feature_model_construction.generate_MIEL_distance_centroidvector(
            df=train_data[relevant_features],
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )

    if len(group_by) > 1:
        train_y_true = np.logical_and(
            train_data[col] == val for col, val in zip(group_by, reference_group_B)
        )
        test_y_true = np.logical_and(
            test_data[col] == val for col, val in zip(group_by, reference_group_B)
        )
    else:
        train_y_true = train_data[group_by] == reference_group_B
        test_y_true = test_data[group_by] == reference_group_B

    y_score = train_MIEL_distance["MIEL_distance"].values.reshape(-1, 1)

    fpr, tpr, thresholds = roc_curve(
        y_true=train_y_true,
        y_score=y_score,
    )

    auc = roc_auc_score(train_y_true, y_score)

    if auc < 1:
        threshold = thresholds[np.argmin((tpr - 1) ** 2 + fpr**2)]
    else:
        threshold = (y_score[train_y_true].min() + y_score[~train_y_true].max()) / 2

    train_y_pred = y_score > threshold

    train_accuracy = accuracy_score(train_y_true, train_y_pred)

    train_confusion = pd.Series(
        data=confusion_matrix(train_y_true, train_y_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    test_MIEL_distance = (
        feature_model_construction.generate_MIEL_distance_centroidvector(
            df=test_data[relevant_features],
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )

    y_score = test_MIEL_distance["MIEL_distance"].values.reshape(-1, 1)

    test_y_pred = y_score > threshold

    test_accuracy = accuracy_score(test_y_true, test_y_pred)

    test_confusion = pd.Series(
        data=confusion_matrix(test_y_true, test_y_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    result = pd.Series(
        {
            "auc": auc,
            "threshold": threshold,
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
        }
    )

    test_middleage_MIEL_distance = (
        feature_model_construction.generate_MIEL_distance_centroidvector(
            df=test_data_middlage[relevant_features],
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )

    train_MIEL_distance.loc[:, ["Sample", "Age", "ExperimentalCondition"]] = train_data[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    test_MIEL_distance.loc[:, ["Sample", "Age", "ExperimentalCondition"]] = test_data[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    test_middleage_MIEL_distance.loc[
        :, ["Sample", "Age", "ExperimentalCondition"]
    ] = test_data_middlage[["Sample", "Age", "ExperimentalCondition"]]

    # HEAT CALCULATIONS BEGIN

    test_cv_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(test_cv[relevant_features]),
        columns=relevant_features,
        index=test_cv.index,
    )
    # test_cv_norm = test_cv[relevant_features]
    test_cv_norm.loc[:, group_by] = test_cv[group_by]

    # HEAT
    for col, refA, refB in zip(group_by, (reference_group_A,), (reference_group_B,)):
        test_cv_youngold = test_cv_norm.loc[test_cv_norm[col].isin([refA, refB]), :]
    corr = test_cv_youngold.corrwith(test_MIEL_distance["MIEL_distance"]).fillna(0)
    corrnorm = corr / np.linalg.norm(corr.values)  # .abs()
    test_heat = test_cv_norm[relevant_features].dot(corrnorm)

    # # EPIAGEHEAT
    # epiAgeVecNorm = (epiAgeVec / np.linalg.norm(epiAgeVec.values)).abs()
    # test_heat = test_cv_norm[relevant_features].dot(epiAgeVecNorm)

    test_heat.name = "HEAT"
    test_heat = test_heat.to_frame()

    test_heat.loc[:, ["Sample", "Age", "ExperimentalCondition"]] = test_cv[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    for col, refA, refB in zip(group_by, (reference_group_A,), (reference_group_B,)):
        test_heat_middleage = test_heat.loc[~test_heat[col].isin([refA, refB]), :]
        test_heat = test_heat.loc[test_heat[col].isin([refA, refB]), :]
    test_heat.set_index(["Sample", "Age", "ExperimentalCondition"], inplace=True)
    test_heat_middleage.set_index(
        ["Sample", "Age", "ExperimentalCondition"], inplace=True
    )

    return (
        result,
        train_confusion,
        test_confusion,
        epiAgeVec,
        corr,
        train_MIEL_distance,
        test_MIEL_distance,
        test_middleage_MIEL_distance,
        test_heat,
        test_heat_middleage,
        group_sizes,
    )


def chromage_with_accuracy_single_cell_middleage(
    data,
    group_by,
    reference_group_A,
    reference_group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting chromage_with_accuracy using seed: {seed}")
    test_size = 0.25
    train_size = 0.75

    relevant_features = data.columns[data.columns.str.contains(subset)]
    data_grouped = data.groupby(["Sample", "ExperimentalCondition"])
    data_middlage = data
    for col, refA, refB in zip(group_by, reference_group_A, reference_group_B):
        data_middlage = data_middlage.loc[~data_middlage[col].isin([refA, refB]), :]
    data = None
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    test_grouped = test_data.groupby(["Sample", "ExperimentalCondition"])

    data_groups = list(data_grouped.groups.keys())
    test_groups = list(test_grouped.groups.keys())
    if any([i not in data_groups for i in test_groups]):
        raise ValueError("Not enough cells to continue")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")
    for col, refA, refB in zip(group_by, reference_group_A, reference_group_B):
        train_data = train_data.loc[train_data[col].isin([refA, refB]), :]
        test_data = test_data.loc[test_data[col].isin([refA, refB]), :]

    train_data, train_group_sizes = bootstrap_data(
        data=train_data,
        subset=subset,
        group_by=["Sample", "ExperimentalCondition"],
        metric="mean",
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )

    test_data, test_group_sizes = bootstrap_data(
        data=test_data,
        subset=subset,
        group_by=["Sample", "ExperimentalCondition"],
        metric="mean",
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )
    group_sizes = []
    for n, d in zip(["train", "test"], [train_group_sizes, test_group_sizes]):
        d = d.to_frame().rename({0: "n"}, axis=1)
        d["split"] = n
        group_sizes.append(d)
    group_sizes = pd.concat(group_sizes, axis=0)

    df_groupby = train_data.groupby(group_by, as_index=True)[relevant_features]
    if len(group_by) == 1:
        reference_group_A = reference_group_A[0]
        reference_group_B = reference_group_B[0]
    A_centroid = df_groupby.get_group(reference_group_A).mean()
    A_centroid.attrs["name"] = f"{tools.join_iterable(reference_group_A)} centroid"
    B_centroid = df_groupby.get_group(reference_group_B).mean()
    B_centroid.attrs["name"] = f"{tools.join_iterable(reference_group_B)} centroid"

    epiAgeVec = B_centroid - A_centroid

    train_MIEL_distance = (
        feature_model_construction.generate_MIEL_distance_centroidvector(
            df=train_data[relevant_features],
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )

    if len(group_by) > 1:
        train_y_true = np.logical_and(
            train_data[col] == val for col, val in zip(group_by, reference_group_B)
        )
        test_y_true = np.logical_and(
            test_data[col] == val for col, val in zip(group_by, reference_group_B)
        )
    else:
        train_y_true = train_data[group_by] == reference_group_B
        test_y_true = test_data[group_by] == reference_group_B

    y_score = train_MIEL_distance["MIEL_distance"].values.reshape(-1, 1)

    fpr, tpr, thresholds = roc_curve(
        y_true=train_y_true,
        y_score=y_score,
    )

    auc = roc_auc_score(train_y_true, y_score)

    if auc < 1:
        threshold = thresholds[np.argmin((tpr - 1) ** 2 + fpr**2)]
    else:
        threshold = (y_score[train_y_true].min() + y_score[~train_y_true].max()) / 2

    train_y_pred = y_score > threshold

    train_accuracy = accuracy_score(train_y_true, train_y_pred)

    train_confusion = pd.Series(
        data=confusion_matrix(train_y_true, train_y_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    test_MIEL_distance = (
        feature_model_construction.generate_MIEL_distance_centroidvector(
            df=test_data[relevant_features],
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )

    y_score = test_MIEL_distance["MIEL_distance"].values.reshape(-1, 1)

    test_y_pred = y_score > threshold

    test_accuracy = accuracy_score(test_y_true, test_y_pred)

    test_confusion = pd.Series(
        data=confusion_matrix(test_y_true, test_y_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    result = pd.Series(
        {
            "auc": auc,
            "threshold": threshold,
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
        }
    )

    test_middleage_MIEL_distance = (
        feature_model_construction.generate_MIEL_distance_centroidvector(
            df=data_middlage[relevant_features],
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )

    train_MIEL_distance[["Sample", "Age", "ExperimentalCondition"]] = train_data[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    test_MIEL_distance[["Sample", "Age", "ExperimentalCondition"]] = test_data[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    test_middleage_MIEL_distance[
        ["cellidx", "Sample", "Age", "ExperimentalCondition"]
    ] = data_middlage[["cellidx", "Sample", "Age", "ExperimentalCondition"]]

    return (
        result,
        train_confusion,
        test_confusion,
        epiAgeVec,
        train_MIEL_distance,
        test_MIEL_distance,
        test_middleage_MIEL_distance,
        group_sizes,
    )


def chromage_with_accuracy_linear_regression(
    data,
    group_by,
    reference_group_A,
    reference_group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting chromage_with_accuracy using seed: {seed}")
    test_size = 0.25
    train_size = 0.75

    relevant_features = data.columns[data.columns.str.contains(subset)]
    data_grouped = data.groupby(["Sample", "ExperimentalCondition"])
    data_middlage = data
    for col, refA, refB in zip(group_by, reference_group_A, reference_group_B):
        data_middlage = data_middlage.loc[~data_middlage[col].isin([refA, refB]), :]
    data = None
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    test_grouped = test_data.groupby(["Sample", "ExperimentalCondition"])

    data_groups = list(data_grouped.groups.keys())
    test_groups = list(test_grouped.groups.keys())
    if any([i not in data_groups for i in test_groups]):
        raise ValueError("Not enough cells to continue")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")
    for col, refA, refB in zip(group_by, reference_group_A, reference_group_B):
        train_data = train_data.loc[train_data[col].isin([refA, refB]), :]
        test_data = test_data.loc[test_data[col].isin([refA, refB]), :]

    train_data, train_group_sizes = bootstrap_data(
        data=train_data,
        subset=subset,
        group_by=["Sample", "ExperimentalCondition"],
        metric="mean",
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )

    test_data, test_group_sizes = bootstrap_data(
        data=test_data,
        subset=subset,
        group_by=["Sample", "ExperimentalCondition"],
        metric="mean",
        num_cells=num_cells,
        num_bootstraps=num_bootstraps,
        seed=seed,
    )
    group_sizes = []
    for n, d in zip(["train", "test"], [train_group_sizes, test_group_sizes]):
        d = d.to_frame().rename({0: "n"}, axis=1)
        d["split"] = n
        group_sizes.append(d)
    group_sizes = pd.concat(group_sizes, axis=0)

    if len(group_by) > 1:
        train_y_true = np.logical_and(
            train_data[col] == val for col, val in zip(group_by, reference_group_B)
        )
        test_y_true = np.logical_and(
            test_data[col] == val for col, val in zip(group_by, reference_group_B)
        )
    else:
        train_y_true = train_data[group_by] == reference_group_B
        test_y_true = test_data[group_by] == reference_group_B

    regressor = LogisticRegression(penalty="l2")
    regressor.fit(train_data[relevant_features], train_y_true)

    train_y_pred = regressor.predict(train_data[relevant_features])

    train_MIEL_distance = pd.DataFrame(
        regressor.predict_proba(train_data[relevant_features]),
        index=train_data.index,
        columns=regressor.classes_,
    )

    # y_score = train_MIEL_distance["MIEL_distance"].values.reshape(-1, 1)

    # fpr, tpr, thresholds = roc_curve(
    #     y_true=train_y_true,
    #     y_score=y_score,
    # )

    # auc = roc_auc_score(train_y_true, y_score)

    # if auc < 1:
    #     threshold = thresholds[np.argmin((tpr - 1) ** 2 + fpr**2)]
    # else:
    #     threshold = (y_score[train_y_true].min() + y_score[~train_y_true].max()) / 2

    # train_y_pred = y_score > threshold

    train_accuracy = accuracy_score(train_y_true, train_y_pred)

    train_confusion = pd.Series(
        data=confusion_matrix(train_y_true.values, train_y_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    test_y_pred = regressor.predict(test_data[relevant_features])

    test_MIEL_distance = pd.DataFrame(
        regressor.predict_proba(test_data[relevant_features]),
        index=test_data.index,
        columns=regressor.classes_,
    )

    # test_MIEL_distance = (
    #     feature_model_construction.generate_MIEL_distance_centroidvector(
    #         df=test_data[relevant_features],
    #         A_centroid=A_centroid,
    #         B_centroid=B_centroid,
    #     )
    # )

    # y_score = test_MIEL_distance["MIEL_distance"].values.reshape(-1, 1)

    # test_y_pred = y_score > threshold

    test_accuracy = accuracy_score(test_y_true, test_y_pred)

    test_confusion = pd.Series(
        data=confusion_matrix(test_y_true.values, test_y_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    result = pd.Series(
        {
            # "auc": auc,
            # "threshold": threshold,
            "auc": None,
            "threshold": None,
            "test_accuracy": test_accuracy,
            "train_accuracy": train_accuracy,
        }
    )

    # test_middleage_MIEL_distance = (
    #     feature_model_construction.generate_MIEL_distance_centroidvector(
    #         df=data_middlage[relevant_features],
    #         A_centroid=A_centroid,
    #         B_centroid=B_centroid,
    #     )
    # )
    test_middleage_MIEL_distance = pd.DataFrame(
        regressor.predict_proba(data_middlage[relevant_features]),
        index=data_middlage.index,
        columns=regressor.classes_,
    )

    train_MIEL_distance[["Sample", "Age", "ExperimentalCondition"]] = train_data[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    test_MIEL_distance[["Sample", "Age", "ExperimentalCondition"]] = test_data[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    test_middleage_MIEL_distance[
        ["Sample", "Age", "ExperimentalCondition"]
    ] = data_middlage[["Sample", "Age", "ExperimentalCondition"]]

    train_MIEL_distance.rename(
        columns={True: "prob_old", False: "prob_young"}, inplace=True
    )
    test_MIEL_distance.rename(
        columns={True: "prob_old", False: "prob_young"}, inplace=True
    )
    test_middleage_MIEL_distance.rename(
        columns={True: "prob_old", False: "prob_young"}, inplace=True
    )
    return (
        result,
        train_confusion,
        test_confusion,
        # epiAgeVec,
        train_MIEL_distance,
        test_MIEL_distance,
        test_middleage_MIEL_distance,
        group_sizes,
    )


from sklearn.decomposition import PCA
import time
from tqdm import tqdm


def s1_o1_chromage_with_accuracy_and_heat(
    data,
    sample_col,
    group_col,
    group_A,
    group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting chromage_with_accuracy using seed: {seed}")
    # data = data.groupby(["Sample", "ExperimentalCondition"]).sample(200)

    # num_bootstraps = 100
    # test_size = 0.25
    # train_size = 0.75

    # subset = "TXT_TAS"
    relevant_features = data.columns[data.columns.str.contains(subset)]

    # tmpmodel = feature_model_construction.ChromAgeModel()
    # tmpdf = data.groupby(["Sample", "ExperimentalCondition"]).mean().reset_index()
    # # tmpzscore = zscore_data(data=tmpdf, group_by=None, subset=subset)
    # # tmpdf[relevant_features] = tmpzscore[relevant_features]
    # tmpdffit = tmpdf.loc[
    #     tmpdf["ExperimentalCondition"].isin(("Old", "Young")),
    #     # & (~tmpdf["Sample"].isin(("301_c_Adams",))),
    #     :,
    # ]
    # tmpmodel.fit(
    #     data=tmpdffit[[*group_by, *relevant_features]],
    #     group_by=group_by,
    #     group_A=group_A,
    #     group_B=group_B,
    # )
    # ss = tmpmodel.score(
    #     data=tmpdf[[*relevant_features]],
    # ).to_frame()
    # ss[["Sample", "ExperimentalCondition"]] = tmpdf[["Sample", "ExperimentalCondition"]]

    data["agenum"] = data["Age"].str.extract("([0-9]+)").astype(int)

    # sampled = data  # .groupby(["Sample", "ExperimentalCondition"]).sample(200)
    # ss1 = tmpmodel.project_orthogonal_subspace(
    #     data=sampled[[*relevant_features]],
    # )
    # ss1[["Sample", "agenum", "ExperimentalCondition"]] = sampled[
    #     ["Sample", "agenum", "ExperimentalCondition"]
    # ]

    # ss2 = tmpmodel.score(
    #     data=data[[*relevant_features]],
    # ).to_frame()

    # ss2[["Sample", "agenum", "ExperimentalCondition"]] = data[
    #     ["Sample", "agenum", "ExperimentalCondition"]
    # ]

    # ss2.groupby(["Sample", "agenum", "ExperimentalCondition"]).std().to_csv(
    #     "MIEL72_chromage_std.csv"
    # )
    # ss2.groupby(["Sample", "agenum", "ExperimentalCondition"]).mean().to_csv(
    #     "MIEL72_chromage_mean.csv"
    # )

    # ss3 = tmpmodel.score_orthogonal(
    #     data=data[[*relevant_features]],
    # ).to_frame()

    # ss3[["Sample", "agenum", "ExperimentalCondition"]] = data[
    #     ["Sample", "agenum", "ExperimentalCondition"]
    # ]

    zscore = zscore_data(data=data, group_by=None, subset=subset)
    new = data.columns[~data.columns.isin(zscore.columns)]
    zscore = pd.concat([zscore, data.loc[:, new]], axis=1)
    scdata = zscore

    # scdata = data

    chromagemodel = feature_model_construction.ChromAgeModel()
    chromagemodel.fit(
        data=scdata[[sample_col, group_col, *relevant_features]],
        sample_col=sample_col,
        group_col=group_col,
        feature_cols=relevant_features,
        group_A=group_A,
        group_B=group_B,
    )

    sc_chromage = chromagemodel.score(
        data=scdata[[*relevant_features]],
    ).to_frame()

    sc_chromage[["Sample", "agenum", "ExperimentalCondition"]] = scdata[
        ["Sample", "agenum", "ExperimentalCondition"]
    ]

    sc_orthogonal = chromagemodel.score_orthogonal(scdata[relevant_features]).to_frame()
    sc_orthogonal[["Sample", "Age", "ExperimentalCondition"]] = scdata[
        ["Sample", "Age", "ExperimentalCondition"]
    ]
    # sc_orthogonal_grouped = sc_orthogonal.groupby(["Sample", "Age", "ExperimentalCondition"])
    # sc_orthogonal = pd.concat([sc_orthogonal_grouped.mean(), sc_orthogonal_grouped.std()], axis=1)
    # sc_orthogonal.columns = ["ChromAgeOrthogonal_mean", "ChromAgeOrthogonal_stdev"]

    start = time.perf_counter()
    sc_orthogonal_vectorspace = chromagemodel.project_orthogonal_subspace(
        data=scdata[[*relevant_features]],
    )
    sc_orthogonal_vectorspace[["Sample", "agenum", "ExperimentalCondition"]] = scdata[
        ["Sample", "agenum", "ExperimentalCondition"]
    ]
    stop = time.perf_counter()
    print((stop - start) / 60)

    totalgrp = sc_chromage.groupby("ExperimentalCondition")
    yo_score = pd.concat([totalgrp.get_group("Young"), totalgrp.get_group("Old")])
    yo_pred = yo_score["ChromAgeDistance"] > chromagemodel.threshold
    yo_true = yo_score["ExperimentalCondition"] == "Old"

    accuracy = pd.Series([accuracy_score(yo_true, yo_pred)], index=["accuracy"])
    confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    # PCA Analysis
    pca = PCA(n_components=100)
    sc_ortho_pca = pd.DataFrame(
        pca.fit_transform(
            # sc_orthogonal_vectorspace[relevant_features]
            StandardScaler().fit_transform(sc_orthogonal_vectorspace[relevant_features])
        ),
        index=sc_orthogonal_vectorspace.index,
    )
    sc_ortho_pca[
        ["Sample", "agenum", "ExperimentalCondition"]
    ] = sc_orthogonal_vectorspace[["Sample", "agenum", "ExperimentalCondition"]]
    sc_ortho_pca_projection = pd.concat(
        [
            sc_orthogonal_vectorspace[relevant_features].dot(pc)
            for pc in pca.components_
        ],
        axis=1,
    )
    sc_ortho_pca_projection[
        ["Sample", "agenum", "ExperimentalCondition"]
    ] = sc_orthogonal_vectorspace[["Sample", "agenum", "ExperimentalCondition"]]
    # sc_ortho_pca.groupby(["Sample", "agenum", "ExperimentalCondition"]).std().iloc[:,4:].sum(axis=1)
    # pcnorm = sc_ortho_pca.set_index(["Sample", "agenum", "ExperimentalCondition"]).apply(lambda x: np.linalg.norm(x.iloc[4:]), axis=1)
    # dfpcstd = sc_ortho_pca.groupby(["Sample", "agenum", "ExperimentalCondition"]).std()
    # dfpcstd.reset_index().corrwith(dfpcstd.reset_index()["agenum"])

    start = time.perf_counter()
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    boot_accuracy = []
    boot_confusion = []
    boot_chromage = []
    sc_chromage_grouped = sc_chromage.groupby(
        ["Sample", "agenum", "ExperimentalCondition"]
    )
    for b in tqdm(range(num_bootstraps)):
        # for boot_idx in tqdm(
        #     rng.choice(
        #         scdata.index.values, size=(num_bootstraps, scdata.shape[0]), replace=True
        #     )
        # ):
        boot_data = sc_chromage_grouped.sample(frac=1, replace=True, random_state=rng)
        chromage_grouped = boot_data.groupby(
            ["Sample", "agenum", "ExperimentalCondition"]
        )
        # boot_chromage_ = pd.concat([chromage_grouped.mean(), chromage_grouped.std()], axis=1)
        # boot_chromage_.columns = ["ChromAgeDistance_mean", "ChromAgeDistance_stdev"]
        boot_chromage.append(chromage_grouped.mean())
        chromage_grouped = boot_data.groupby("ExperimentalCondition")
        yo_score = pd.concat(
            [chromage_grouped.get_group("Young"), chromage_grouped.get_group("Old")]
        )
        yo_pred = yo_score["ChromAgeDistance"] > chromagemodel.threshold
        yo_true = yo_score["ExperimentalCondition"] == "Old"
        boot_accuracy.append(accuracy_score(yo_true, yo_pred))
        boot_confusion.append(
            pd.Series(
                data=confusion_matrix(yo_true, yo_pred).ravel(),
                index=["true_neg", "false_pos", "false_neg", "true_pos"],
            )
        )
    stop = time.perf_counter()
    print((stop - start) / 60)
    boot_chromage = pd.concat(boot_chromage, axis=1)
    boot_chromage = pd.concat(
        [boot_chromage.mean(axis=1), boot_chromage.std(axis=1)], axis=1
    )
    boot_chromage.columns = ["ChromAgeDistance_mean", "ChromAgeDistance_stdev"]
    boot_accuracy = pd.Series(
        [np.mean(boot_accuracy), np.std(boot_accuracy)],
        index=["accuracy_mean", "accuracy_stdev"],
    )
    boot_confusion = pd.concat(boot_confusion, axis=1)
    boot_confusion = pd.concat(
        [boot_confusion.mean(axis=1), boot_confusion.std(axis=1)], axis=1
    )
    boot_confusion.columns = ["confusion_mean", "confusion_stdev"]

    return (
        sc_chromage,
        sc_orthogonal,
        sc_orthogonal_vectorspace,
        pca,
        sc_ortho_pca,
        sc_ortho_pca_projection,
        accuracy,
        confusion,
        boot_chromage,
        boot_accuracy,
        boot_confusion,
    )


def s1_o1_chromage_with_accuracy_and_heat_traintestsplit(
    data,
    sample_col,
    group_col,
    group_A,
    group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting chromage_with_accuracy using seed: {seed}")
    # data = data.groupby(["Sample", "ExperimentalCondition"]).sample(200)

    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # subset = "TXT_TAS"
    relevant_features = data.columns[data.columns.str.contains(subset)]

    # tmpmodel = feature_model_construction.ChromAgeModel()
    # tmpdf = data.groupby(["Sample", "ExperimentalCondition"]).mean().reset_index()
    # # tmpzscore = zscore_data(data=tmpdf, group_by=None, subset=subset)
    # # tmpdf[relevant_features] = tmpzscore[relevant_features]
    # tmpdffit = tmpdf.loc[
    #     tmpdf["ExperimentalCondition"].isin(("Old", "Young")),
    #     # & (~tmpdf["Sample"].isin(("301_c_Adams",))),
    #     :,
    # ]
    # tmpmodel.fit(
    #     data=tmpdffit[[*group_by, *relevant_features]],
    #     group_by=group_by,
    #     group_A=group_A,
    #     group_B=group_B,
    # )
    # ss = tmpmodel.score(
    #     data=tmpdf[[*relevant_features]],
    # ).to_frame()
    # ss[["Sample", "ExperimentalCondition"]] = tmpdf[["Sample", "ExperimentalCondition"]]

    data["agenum"] = data["Age"].str.extract("([0-9]+)").astype(int)

    # sampled = data  # .groupby(["Sample", "ExperimentalCondition"]).sample(200)
    # ss1 = tmpmodel.project_orthogonal_subspace(
    #     data=sampled[[*relevant_features]],
    # )
    # ss1[["Sample", "agenum", "ExperimentalCondition"]] = sampled[
    #     ["Sample", "agenum", "ExperimentalCondition"]
    # ]

    # ss2 = tmpmodel.score(
    #     data=data[[*relevant_features]],
    # ).to_frame()

    # ss2[["Sample", "agenum", "ExperimentalCondition"]] = data[
    #     ["Sample", "agenum", "ExperimentalCondition"]
    # ]

    # ss2.groupby(["Sample", "agenum", "ExperimentalCondition"]).std().to_csv(
    #     "MIEL72_chromage_std.csv"
    # )
    # ss2.groupby(["Sample", "agenum", "ExperimentalCondition"]).mean().to_csv(
    #     "MIEL72_chromage_mean.csv"
    # )

    # ss3 = tmpmodel.score_orthogonal(
    #     data=data[[*relevant_features]],
    # ).to_frame()

    # ss3[["Sample", "agenum", "ExperimentalCondition"]] = data[
    #     ["Sample", "agenum", "ExperimentalCondition"]
    # ]

    zscore = zscore_data(data=data, group_by=None, subset=subset)
    new = data.columns[~data.columns.isin(zscore.columns)]
    zscore = pd.concat([zscore, data.loc[:, new]], axis=1)
    scdata = zscore

    # scdata = data

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)
    test_grouped = test_data.groupby(["Sample", "ExperimentalCondition"])

    scdata.loc[test_data.index, "split"] = "test"
    scdata.loc[train_data.index, "split"] = "train"

    data_groups = list(data_grouped.groups.keys())
    test_groups = list(test_grouped.groups.keys())
    if any([i not in data_groups for i in test_groups]):
        raise ValueError("Not enough cells to continue")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")

    chromagemodel = feature_model_construction.ChromAgeModel()
    chromagemodel.fit(
        data=train_data[[sample_col, group_col, *relevant_features]],
        sample_col=sample_col,
        group_col=group_col,
        feature_cols=relevant_features,
        group_A=group_A,
        group_B=group_B,
    )

    # Full Data
    sc_chromage = chromagemodel.score(
        data=scdata[[*relevant_features]],
    ).to_frame()
    sc_chromage[["Sample", "agenum", "ExperimentalCondition", "split"]] = scdata[
        ["Sample", "agenum", "ExperimentalCondition", "split"]
    ]

    sc_train_chromage = sc_chromage.loc[sc_chromage["split"] == "train"]
    sc_test_chromage = sc_chromage.loc[sc_chromage["split"] == "test"]

    # sc_orthogonal = chromagemodel.score_orthogonal(scdata[relevant_features]).to_frame()
    # sc_orthogonal[["Sample", "Age", "ExperimentalCondition", "split"]] = scdata[
    #     ["Sample", "Age", "ExperimentalCondition", "split"]
    # ]
    # # sc_orthogonal_grouped = sc_orthogonal.groupby(["Sample", "Age", "ExperimentalCondition"])
    # # sc_orthogonal = pd.concat([sc_orthogonal_grouped.mean(), sc_orthogonal_grouped.std()], axis=1)
    # # sc_orthogonal.columns = ["ChromAgeOrthogonal_mean", "ChromAgeOrthogonal_stdev"]

    # start = time.perf_counter()
    # sc_orthogonal_vectorspace = chromagemodel.project_orthogonal_subspace(
    #     data=scdata[[*relevant_features]],
    # )
    # sc_orthogonal_vectorspace[
    #     ["Sample", "agenum", "ExperimentalCondition", "split"]
    # ] = scdata[["Sample", "agenum", "ExperimentalCondition", "split"]]
    # stop = time.perf_counter()
    # print((stop - start) / 60)

    # totalgrp = sc_chromage.groupby("ExperimentalCondition")
    # yo_score = pd.concat([totalgrp.get_group("Young"), totalgrp.get_group("Old")])
    # yo_pred = yo_score["ChromAgeDistance"] > chromagemodel.threshold
    # yo_true = yo_score["ExperimentalCondition"] == "Old"

    # accuracy = pd.Series([accuracy_score(yo_true, yo_pred)], index=["accuracy"])
    # confusion = pd.Series(
    #     data=confusion_matrix(yo_true, yo_pred).ravel(),
    #     index=["true_neg", "false_pos", "false_neg", "true_pos"],
    # )

    # # PCA Analysis
    # pca = PCA(n_components=100)
    # sc_ortho_pca = pd.DataFrame(
    #     pca.fit_transform(
    #         # sc_orthogonal_vectorspace[relevant_features]
    #         StandardScaler().fit_transform(sc_orthogonal_vectorspace[relevant_features])
    #     ),
    #     index=sc_orthogonal_vectorspace.index,
    # )
    # sc_ortho_pca[
    #     ["Sample", "agenum", "ExperimentalCondition", "split"]
    # ] = sc_orthogonal_vectorspace[
    #     ["Sample", "agenum", "ExperimentalCondition", "split"]
    # ]
    # sc_ortho_pca_projection = pd.concat(
    #     [
    #         sc_orthogonal_vectorspace[relevant_features].dot(pc)
    #         for pc in pca.components_
    #     ],
    #     axis=1,
    # )
    # sc_ortho_pca_projection[
    #     ["Sample", "agenum", "ExperimentalCondition", "split"]
    # ] = sc_orthogonal_vectorspace[
    #     ["Sample", "agenum", "ExperimentalCondition", "split"]
    # ]
    # # sc_ortho_pca.groupby(["Sample", "agenum", "ExperimentalCondition"]).std().iloc[:,4:].sum(axis=1)
    # # pcnorm = sc_ortho_pca.set_index(["Sample", "agenum", "ExperimentalCondition"]).apply(lambda x: np.linalg.norm(x.iloc[4:]), axis=1)
    # # dfpcstd = sc_ortho_pca.groupby(["Sample", "agenum", "ExperimentalCondition"]).std()
    # # dfpcstd.reset_index().corrwith(dfpcstd.reset_index()["agenum"])

    start = time.perf_counter()
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    boot_train_accuracy = []
    boot_train_confusion = []
    boot_train_chromage = []
    boot_test_accuracy = []
    boot_test_confusion = []
    boot_test_chromage = []
    boot_accuracy_curve = []
    sc_train_chromage_grouped = sc_train_chromage.groupby(
        ["Sample", "agenum", "ExperimentalCondition"]
    )
    sc_test_chromage_grouped = sc_test_chromage.groupby(
        ["Sample", "agenum", "ExperimentalCondition"]
    )
    for b in tqdm(range(num_bootstraps)):
        boot_data = sc_train_chromage_grouped.sample(
            frac=1, replace=True, random_state=rng
        )
        chromage_grouped = boot_data.groupby(
            ["Sample", "agenum", "ExperimentalCondition"]
        )
        # boot_chromage_ = pd.concat([chromage_grouped.mean(), chromage_grouped.std()], axis=1)
        # boot_chromage_.columns = ["ChromAgeDistance_mean", "ChromAgeDistance_stdev"]
        boot_train_chromage.append(chromage_grouped["ChromAgeDistance"].mean())
        chromage_grouped = boot_data.groupby("ExperimentalCondition")
        yo_score = pd.concat(
            [chromage_grouped.get_group("Young"), chromage_grouped.get_group("Old")]
        )
        yo_pred = yo_score["ChromAgeDistance"] > chromagemodel.threshold
        yo_true = yo_score["ExperimentalCondition"] == "Old"
        boot_train_accuracy.append(accuracy_score(yo_true, yo_pred))
        boot_train_confusion.append(
            pd.Series(
                data=confusion_matrix(yo_true, yo_pred).ravel(),
                index=["true_neg", "false_pos", "false_neg", "true_pos"],
            )
        )

        boot_data = sc_test_chromage_grouped.sample(
            frac=1, replace=True, random_state=rng
        )
        chromage_grouped = boot_data.groupby(
            ["Sample", "agenum", "ExperimentalCondition"]
        )
        # boot_chromage_ = pd.concat([chromage_grouped.mean(), chromage_grouped.std()], axis=1)
        # boot_chromage_.columns = ["ChromAgeDistance_mean", "ChromAgeDistance_stdev"]
        boot_test_chromage.append(chromage_grouped["ChromAgeDistance"].mean())
        chromage_grouped = boot_data.groupby("ExperimentalCondition")
        yo_score = pd.concat(
            [chromage_grouped.get_group("Young"), chromage_grouped.get_group("Old")]
        )
        yo_pred = yo_score["ChromAgeDistance"] > chromagemodel.threshold
        yo_true = yo_score["ExperimentalCondition"] == "Old"
        boot_test_accuracy.append(accuracy_score(yo_true, yo_pred))
        boot_test_confusion.append(
            pd.Series(
                data=confusion_matrix(yo_true, yo_pred).ravel(),
                index=["true_neg", "false_pos", "false_neg", "true_pos"],
            )
        )
        acc_curve = []
        grp_sizes = sc_test_chromage_grouped.size()
        max_size = 1000 if grp_sizes.min() > 1000 else grp_sizes.min()
        acc_bin_sizes = [1, *np.logspace(1, 3, base=10, num=20).astype(int)]
        # start = time.perf_counter()
        for s in acc_bin_sizes:
            boot_yo_score = (
                sc_test_chromage_grouped.sample(s, replace=True, random_state=rng)
                .groupby(["Sample", "agenum", "ExperimentalCondition"])[
                    "ChromAgeDistance"
                ]
                .mean()
                .reset_index()
            )
            boot_yo_score = boot_yo_score[
                boot_yo_score["ExperimentalCondition"].isin(["Young", "Old"])
            ]
            boot_yo_pred = boot_yo_score["ChromAgeDistance"] > chromagemodel.threshold
            boot_yo_true = boot_yo_score["ExperimentalCondition"] == "Old"
            acc_curve.append(accuracy_score(boot_yo_true, boot_yo_pred))
        # stop = time.perf_counter()
        # print((stop - start) / 60)
        boot_accuracy_curve.append(pd.Series(acc_curve, index=acc_bin_sizes))
    stop = time.perf_counter()
    print((stop - start) / 60)
    boot_train_chromage = pd.concat(boot_train_chromage, axis=1)
    boot_train_chromage = pd.concat(
        [boot_train_chromage.mean(axis=1), boot_train_chromage.std(axis=1)], axis=1
    )
    boot_train_chromage.columns = ["ChromAgeDistance_mean", "ChromAgeDistance_stdev"]
    boot_train_accuracy = pd.Series(
        [np.mean(boot_train_accuracy), np.std(boot_train_accuracy)],
        index=["accuracy_mean", "accuracy_stdev"],
    )
    boot_train_confusion = pd.concat(boot_train_confusion, axis=1)
    boot_train_confusion = pd.concat(
        [boot_train_confusion.mean(axis=1), boot_train_confusion.std(axis=1)], axis=1
    )
    boot_train_confusion.columns = ["confusion_mean", "confusion_stdev"]

    boot_test_chromage = pd.concat(boot_test_chromage, axis=1)
    boot_test_chromage = pd.concat(
        [boot_test_chromage.mean(axis=1), boot_test_chromage.std(axis=1)], axis=1
    )
    boot_test_chromage.columns = ["ChromAgeDistance_mean", "ChromAgeDistance_stdev"]
    boot_test_accuracy = pd.Series(
        [np.mean(boot_test_accuracy), np.std(boot_test_accuracy)],
        index=["accuracy_mean", "accuracy_stdev"],
    )
    boot_test_confusion = pd.concat(boot_test_confusion, axis=1)
    boot_test_confusion = pd.concat(
        [boot_test_confusion.mean(axis=1), boot_test_confusion.std(axis=1)], axis=1
    )
    boot_test_confusion.columns = ["confusion_mean", "confusion_stdev"]

    boot_accuracy_curve = pd.concat(boot_accuracy_curve, axis=1)
    boot_accuracy_curve = pd.concat(
        [boot_accuracy_curve.mean(axis=1), boot_accuracy_curve.std(axis=1)], axis=1
    )
    boot_accuracy_curve.columns = ["accuracy_curve_mean", "accuracy_curve_stdev"]

    return (
        sc_chromage,
        sc_orthogonal,
        sc_orthogonal_vectorspace,
        pca,
        sc_ortho_pca,
        sc_ortho_pca_projection,
        accuracy,
        confusion,
        boot_train_chromage,
        boot_train_accuracy,
        boot_train_confusion,
        boot_test_chromage,
        boot_test_accuracy,
        boot_test_confusion,
        boot_accuracy_curve,
    )


def s1_o1_chromage_cenvec_bootstrap_traintestsplit(
    scdata,
    sample_col,
    group_col,
    group_A,
    group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting chromage_with_accuracy using seed: {seed}")
    # scdata = scdata.groupby(["Sample", "ExperimentalCondition"]).sample(200)

    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # boot_accuracy_curve = s1_o1_chromage_cenvec_bootstrap_accuracy_curve(
    #     scdata=scdata,
    #     num_bootstraps=num_bootstraps,
    #     seed=seed,
    #     subset=subset,
    # )

    # subset = "TXT_TAS"
    relevant_features = scdata.columns[scdata.columns.str.contains(subset)]

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat.index.values,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scdata.loc[test_data, "split"] = "test"
    scdata.loc[train_data, "split"] = "train"
    assert not any(scdata["split"].isna().values)
    data_grouped = scdata.groupby(
        ["split", "Sample", "agenum", "ExperimentalCondition"],
        as_index=True,
    )
    data_groups = list(data_grouped.groups.keys())
    if len(data_grouped) % 2 != 0:  # add nnonzero
        raise ValueError("Train test split dropped a group")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")
    scdata = None

    rng = np.random.Generator(np.random.PCG64(seed=seed))
    # start = time.perf_counter()
    group_sizes = data_grouped.size()
    boot_data = []
    boot_meta = []
    for group, size in tqdm([*group_sizes.items()], "bootstrapping"):
        if num_cells == "original":
            num_cells = size
        boot_idxs = rng.choice(size, size=(num_bootstraps, num_cells), replace=True)
        temp_data = data_grouped.get_group(group)[relevant_features]
        for i, idxs in enumerate(boot_idxs):
            boot_data.append(temp_data.iloc[idxs].mean().values)
            boot_meta.append(np.array([i + 1, *group]))

    boot_meta = pd.DataFrame(
        np.array(boot_meta),
        columns=[
            "bootstrap",
            "split",
            "Sample",
            "agenum",
            "ExperimentalCondition",
        ],
    )
    boot_data = pd.DataFrame(
        np.array(boot_data),
        columns=relevant_features,
    )
    boot_data = pd.concat([boot_meta, boot_data], axis=1)
    # stop = time.perf_counter()
    # print((stop - start) / 60)

    chromagemodel = feature_model_construction.ChromAgeModel()
    chromagemodel.fit(
        data=boot_data.loc[boot_data["split"] == "train"][
            [sample_col, group_col, *relevant_features]
        ],
        sample_col=sample_col,
        group_col=group_col,
        feature_cols=relevant_features,
        group_A=group_A,
        group_B=group_B,
    )
    boot_chromage_axis = chromagemodel.ChromAgeVec

    # Full Data
    boot_chromage = chromagemodel.score(
        data=boot_data[[*relevant_features]],
    ).to_frame()
    boot_chromage.loc[
        :, "ChromAge_orthogonal_distance"
    ] = chromagemodel.score_orthogonal(
        data=boot_data[[*relevant_features]],
    )
    boot_chromage[["split", "Sample", "agenum", "ExperimentalCondition"]] = boot_data[
        ["split", "Sample", "agenum", "ExperimentalCondition"]
    ]
    chromage_grouped = boot_chromage.groupby(["split", "ExperimentalCondition"])
    # Get train accuracy
    yo_score = pd.concat(
        [
            chromage_grouped.get_group(("train", "Young")),
            chromage_grouped.get_group(("train", "Old")),
        ]
    )
    yo_pred = yo_score["ChromAge"] > chromagemodel.threshold
    yo_true = yo_score["ExperimentalCondition"] == "Old"
    boot_train_accuracy = accuracy_score(yo_true, yo_pred)
    boot_train_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    yo_score = pd.concat(
        [
            chromage_grouped.get_group(("test", "Young")),
            chromage_grouped.get_group(("test", "Old")),
        ]
    )
    yo_pred = yo_score["ChromAge"] > chromagemodel.threshold
    yo_true = yo_score["ExperimentalCondition"] == "Old"
    boot_test_accuracy = accuracy_score(yo_true, yo_pred)
    boot_test_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    return (
        boot_chromage,
        boot_chromage_axis,
        boot_train_accuracy,
        boot_train_confusion,
        boot_test_accuracy,
        boot_test_confusion,
        # boot_accuracy_curve,
        group_sizes,
    )


def s1_o1_chromage_cenvec_bootstrap_accuracy_curve(
    scdata,
    num_bootstraps,
    seed,
    subset: str,
):
    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # subset = "TXT_TAS"
    relevant_features = scdata.columns[scdata.columns.str.contains(subset)]

    # zscore = zscore_data(data=scdata, group_by=None, subset=subset)
    # new = scdata.columns[~scdata.columns.isin(zscore.columns)]
    # zscore = pd.concat([zscore, scdata.loc[:, new]], axis=1)
    # scdata = zscore

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat.index.values,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scdata.loc[test_data, "split"] = "test"
    scdata.loc[train_data, "split"] = "train"
    data_grouped = scdata.groupby(
        ["split", "Sample", "agenum", "ExperimentalCondition"],
        as_index=True,
    )
    data_groups = list(data_grouped.groups.keys())
    if len(data_grouped) % 2 != 0:  # if nonzero
        raise ValueError("Train test split dropped a group")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")
    scdata = None

    rng = np.random.Generator(np.random.PCG64(seed=seed))

    acc_curve = []
    group_sizes = data_grouped.size()
    max_size = group_sizes.min()
    acc_bin_sizes = [
        1,
        *np.logspace(1, np.log10(max_size), base=10, num=25).astype(int),
    ]
    start = time.perf_counter()
    for s in tqdm(acc_bin_sizes, "accuracy curve"):
        boot_data = []
        boot_meta = []
        for group, size in group_sizes.items():
            if not any(i in ["Young", "Old"] for i in group):
                continue
            if s == "original":
                s = size
            boot_idxs = rng.choice(size, size=(num_bootstraps, s), replace=True)
            temp_data = data_grouped.get_group(group)
            for i, idxs in enumerate(boot_idxs):
                temp_boot = temp_data.iloc[idxs][relevant_features].mean()
                boot_data.append(temp_boot)
            boot_meta.append(
                pd.DataFrame(
                    np.array([*group] * num_bootstraps).reshape(
                        (num_bootstraps, len(group))
                    ),
                    columns=["split", "Sample", "agenum", "ExperimentalCondition"],
                )
            )
        boot_data = pd.concat(boot_data, ignore_index=True, axis=1).T
        boot_meta = pd.concat(boot_meta, ignore_index=True, axis=0)
        boot_data = pd.concat([boot_meta, boot_data], axis=1)
        for_training = boot_data["split"] == "train"

        chromagemodel = feature_model_construction.ChromAgeModel()
        chromagemodel.fit(
            data=boot_data.loc[
                for_training, ["Sample", "ExperimentalCondition", *relevant_features]
            ],
            sample_col="Sample",
            group_col="ExperimentalCondition",
            feature_cols=relevant_features,
            group_A="Young",
            group_B="Old",
        )
        boot_chromage = chromagemodel.score(
            data=boot_data.loc[~for_training, relevant_features],
        ).to_frame()
        boot_chromage["ExperimentalCondition"] = boot_data.loc[
            ~for_training, "ExperimentalCondition"
        ]
        boot_yo_pred = boot_chromage["ChromAge"] > chromagemodel.threshold
        boot_yo_true = boot_chromage["ExperimentalCondition"] == "Old"
        acc_curve.append(accuracy_score(boot_yo_true, boot_yo_pred))
    stop = time.perf_counter()
    print((stop - start) / 60)
    boot_accuracy_curve = pd.DataFrame(
        [acc_curve, acc_bin_sizes], index=["accuracy", "bin_size"]
    ).T

    return boot_accuracy_curve


from sklearn.svm import SVC


def s1_o1_chromage_svm_bootstrap_traintestsplit(
    scdata,
    sample_col,
    group_col,
    group_A,
    group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting chromage using seed: {seed}")
    # scdata = scdata.groupby(["Sample", "ExperimentalCondition"]).sample(200)

    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # subset = "TXT_TAS"
    relevant_features = scdata.columns[scdata.columns.str.contains(subset)]

    # zscore = zscore_data(data=scdata, group_by=None, subset=subset)
    # new = scdata.columns[~scdata.columns.isin(zscore.columns)]
    # zscore = pd.concat([zscore, scdata.loc[:, new]], axis=1)
    # scdata = zscore

    # boot_accuracy_curve = s1_o1_chromage_svm_bootstrap_accuracy_curve(
    #     scdata=scdata,
    #     num_bootstraps=num_bootstraps,
    #     seed=seed,
    #     subset=subset,
    # )

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat.index.values,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scdata.loc[test_data, "split"] = "test"
    scdata.loc[train_data, "split"] = "train"
    assert not any(scdata["split"].isna().values)
    data_grouped = scdata.groupby(
        ["split", "Sample", "agenum", "ExperimentalCondition"],
        as_index=True,
    )
    scdata = None
    data_groups = list(data_grouped.groups.keys())
    if len(data_grouped) % 2 != 0:  # add nnonzero
        raise ValueError("Train test split dropped a group")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")

    rng = np.random.Generator(np.random.PCG64(seed=seed))
    # start = time.perf_counter()
    group_sizes = data_grouped.size()
    boot_data = []
    boot_meta = []
    for group, size in tqdm([*group_sizes.items()], "bootstrapping"):
        if num_cells == "original":
            num_cells = size
        boot_idxs = rng.choice(size, size=(num_bootstraps, num_cells), replace=True)
        temp_data = data_grouped.get_group(group)[relevant_features]
        for i, idxs in enumerate(boot_idxs):
            boot_data.append(temp_data.iloc[idxs].mean().values)
            boot_meta.append(np.array([i + 1, *group]))

    boot_meta = pd.DataFrame(
        np.array(boot_meta),
        columns=[
            "bootstrap",
            "split",
            "Sample",
            "agenum",
            "ExperimentalCondition",
        ],
    )
    boot_data = pd.DataFrame(
        np.array(boot_data),
        columns=relevant_features,
    )
    boot_data = pd.concat([boot_meta, boot_data], axis=1)
    # stop = time.perf_counter()
    # print((stop - start) / 60)

    chromagemodel = SVC(
        kernel="linear", verbose=True, class_weight="balanced", probability=False
    )
    for_training = (boot_data["split"] == "train") & boot_data[
        "ExperimentalCondition"
    ].isin(["Young", "Old"])
    chromagemodel.fit(
        X=boot_data.loc[for_training, relevant_features],
        y=boot_data.loc[for_training, group_col] == group_B,
    )
    # Get feature weights
    boot_chromage_axis = pd.Series(
        data=chromagemodel.coef_.squeeze(),
        index=chromagemodel.feature_names_in_,
    )
    # Full Data
    boot_chromage = pd.DataFrame(
        chromagemodel.decision_function(X=boot_data[[*relevant_features]]),
        index=boot_data.index,
        columns=["ChromAge"],
    )
    # Get the hyperplane coefficients and intercept, normalize
    coef_ = chromagemodel.coef_.squeeze()
    coef_l2norm = np.linalg.norm(coef_, ord=2)
    coef_unit = coef_ / coef_l2norm
    b = chromagemodel.intercept_

    def _scalar_projection(data):
        return np.dot(data, coef_) / coef_l2norm

    def _vector_projection(data):
        return np.outer(_scalar_projection(data), coef_unit)

    def _ortho_vector(data):
        return data - _vector_projection(data)

    def _ortho_distance(data):
        return np.linalg.norm(_ortho_vector(data), ord=2)

    # Center plane at 0
    data = boot_data[[*relevant_features]] - b[0]
    # project data onto plane via orthogonal projection
    # calculate centroid
    data_plane_centroid = _ortho_vector(data).mean(axis=0)
    # Center the data at the centroid
    data -= data_plane_centroid
    # Find the distance to the normal of the hyperplane
    chromage_ortho_distance = _ortho_vector(data).apply(np.linalg.norm, axis=1)
    # add to dataframe
    boot_chromage["chromage_orthogonal_distance"] = chromage_ortho_distance
    boot_chromage["chromage_distance"] = boot_chromage["ChromAge"] / coef_l2norm

    boot_chromage[["split", "Sample", "agenum", "ExperimentalCondition"]] = boot_data[
        ["split", "Sample", "agenum", "ExperimentalCondition"]
    ]
    chromage_grouped = boot_chromage.groupby(["split", "ExperimentalCondition"])
    # Get train accuracy
    yo_score = pd.concat(
        [
            chromage_grouped.get_group(("train", "Young")),
            chromage_grouped.get_group(("train", "Old")),
        ]
    )
    yo_pred = yo_score["ChromAge"] > 0
    yo_true = yo_score["ExperimentalCondition"] == "Old"
    boot_train_accuracy = accuracy_score(yo_true, yo_pred)
    boot_train_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    yo_score = pd.concat(
        [
            chromage_grouped.get_group(("test", "Young")),
            chromage_grouped.get_group(("test", "Old")),
        ]
    )
    yo_pred = yo_score["ChromAge"] > 0
    yo_true = yo_score["ExperimentalCondition"] == "Old"
    boot_test_accuracy = accuracy_score(yo_true, yo_pred)
    boot_test_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    return (
        boot_chromage,
        boot_chromage_axis,
        boot_train_accuracy,
        boot_train_confusion,
        boot_test_accuracy,
        boot_test_confusion,
        # boot_accuracy_curve,
        group_sizes,
    )


def s1_o1_chromage_svm_bootstrap_accuracy_curve(
    scdata,
    num_bootstraps,
    seed,
    subset: str,
):
    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # subset = "TXT_TAS"
    relevant_features = scdata.columns[scdata.columns.str.contains(subset)]

    # zscore = zscore_data(data=scdata, group_by=None, subset=subset)
    # new = scdata.columns[~scdata.columns.isin(zscore.columns)]
    # zscore = pd.concat([zscore, scdata.loc[:, new]], axis=1)
    # scdata = zscore

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat.index.values,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scdata.loc[test_data, "split"] = "test"
    scdata.loc[train_data, "split"] = "train"
    data_grouped = scdata.groupby(
        ["split", "Sample", "agenum", "ExperimentalCondition"],
        as_index=True,
    )
    data_groups = list(data_grouped.groups.keys())
    if len(data_grouped) % 2 != 0:  # add nnonzero
        raise ValueError("Train test split dropped a group")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")
    scdata = None

    rng = np.random.Generator(np.random.PCG64(seed=seed))

    acc_curve = []
    group_sizes = data_grouped.size()
    max_size = group_sizes.min()
    acc_bin_sizes = [
        1,
        *np.logspace(1, np.log10(max_size), base=10, num=25).astype(int),
    ]
    start = time.perf_counter()
    for s in tqdm(acc_bin_sizes, "accuracy curve"):
        boot_data = []
        boot_meta = []
        for group, size in group_sizes.items():
            if not any(i in ["Young", "Old"] for i in group):
                continue
            if s == "original":
                s = size
            boot_idxs = rng.choice(size, size=(num_bootstraps, s), replace=True)
            temp_data = data_grouped.get_group(group)
            for i, idxs in enumerate(boot_idxs):
                temp_boot = temp_data.iloc[idxs][relevant_features].mean()
                boot_data.append(temp_boot)
            boot_meta.append(
                pd.DataFrame(
                    np.array([*group] * num_bootstraps).reshape(
                        (num_bootstraps, len(group))
                    ),
                    columns=["split", "Sample", "agenum", "ExperimentalCondition"],
                )
            )
        boot_data = pd.concat(boot_data, ignore_index=True, axis=1).T
        boot_meta = pd.concat(boot_meta, ignore_index=True, axis=0)
        boot_data = pd.concat([boot_meta, boot_data], axis=1)
        chromagemodel = SVC(
            kernel="linear", verbose=True, class_weight="balanced", probability=False
        )
        for_training = boot_data["split"] == "train"
        chromagemodel.fit(
            X=boot_data.loc[for_training, relevant_features],
            y=boot_data.loc[for_training, "ExperimentalCondition"] == "Old",
        )
        boot_chromage = pd.DataFrame(
            chromagemodel.decision_function(
                X=boot_data.loc[~for_training, relevant_features]
            ),
            index=boot_data.loc[~for_training, :].index,
            columns=["ChromAge"],
        )
        boot_chromage["ExperimentalCondition"] = boot_data.loc[
            ~for_training, "ExperimentalCondition"
        ]
        boot_yo_pred = boot_chromage["ChromAge"] > 0
        boot_yo_true = boot_chromage["ExperimentalCondition"] == "Old"
        acc_curve.append(accuracy_score(boot_yo_true, boot_yo_pred))
    stop = time.perf_counter()
    print((stop - start) / 60)
    boot_accuracy_curve = pd.DataFrame(
        [acc_curve, acc_bin_sizes], index=["accuracy", "bin_size"]
    ).T

    return boot_accuracy_curve


def s1_o1_chromage_lr_bootstrap_traintestsplit(
    scdata,
    sample_col,
    group_col,
    group_A,
    group_B,
    num_cells,
    num_bootstraps,
    seed,
    subset: str,
):
    print(f"starting chromage using seed: {seed}")
    # scdata = scdata.groupby(["Sample", "ExperimentalCondition"]).sample(200)

    # num_bootstraps = 100
    test_size = 0.25
    train_size = 0.75

    # subset = "TXT_TAS"
    relevant_features = scdata.columns[scdata.columns.str.contains(subset)]

    # zscore = zscore_data(data=scdata, group_by=None, subset=subset)
    # new = scdata.columns[~scdata.columns.isin(zscore.columns)]
    # zscore = pd.concat([zscore, scdata.loc[:, new]], axis=1)
    # scdata = zscore

    boot_accuracy_curve = pd.DataFrame([None])

    data_grouped = scdata.groupby(["Sample", "ExperimentalCondition"])
    train_data = []
    test_data = []
    for grp, dat in data_grouped:
        grp_train_data, grp_test_data = train_test_split(
            dat.index.values,
            test_size=test_size,
            train_size=train_size,
            random_state=seed,
        )
        train_data.append(grp_train_data)
        test_data.append(grp_test_data)
    grp_train_data = None
    grp_test_data = None
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)

    scdata.loc[test_data, "split"] = "test"
    scdata.loc[train_data, "split"] = "train"
    assert not any(scdata["split"].isna().values)
    data_grouped = scdata.groupby(
        ["split", "Sample", "agenum", "ExperimentalCondition"],
        as_index=True,
    )
    scdata = None
    data_groups = list(data_grouped.groups.keys())
    if len(data_grouped) % 2 != 0:  # add nnonzero
        raise ValueError("Train test split dropped a group")
    # if any([i < num_bootstraps for i in test_grouped.size().values]):
    #     raise ValueError("Not enough cells to continue")

    rng = np.random.Generator(np.random.PCG64(seed=seed))
    # start = time.perf_counter()
    group_sizes = data_grouped.size()
    boot_data = []
    boot_meta = []
    for group, size in tqdm([*group_sizes.items()], "bootstrapping"):
        if num_cells == "original":
            num_cells = size
        boot_idxs = rng.choice(size, size=(num_bootstraps, num_cells), replace=True)
        temp_data = data_grouped.get_group(group)[relevant_features]
        for i, idxs in enumerate(boot_idxs):
            boot_data.append(temp_data.iloc[idxs].mean().values)
            boot_meta.append(np.array([i + 1, *group]))

    boot_meta = pd.DataFrame(
        np.array(boot_meta),
        columns=[
            "bootstrap",
            "split",
            "Sample",
            "agenum",
            "ExperimentalCondition",
        ],
    )
    boot_data = pd.DataFrame(
        np.array(boot_data),
        columns=relevant_features,
    )
    boot_data = pd.concat([boot_meta, boot_data], axis=1)
    # stop = time.perf_counter()
    # print((stop - start) / 60)

    chromagemodel = LogisticRegression(penalty="l2")

    for_training = (boot_data["split"] == "train") & boot_data[
        "ExperimentalCondition"
    ].isin(["Young", "Old"])
    chromagemodel.fit(
        X=boot_data.loc[for_training, relevant_features],
        y=boot_data.loc[for_training, group_col] == group_B,
    )

    # Full Data
    boot_chromage = pd.DataFrame(
        chromagemodel.predict_proba(X=boot_data[[*relevant_features]])[:, 1],
        index=boot_data.index,
        columns=["ChromAge"],
    )
    boot_chromage[["split", "Sample", "agenum", "ExperimentalCondition"]] = boot_data[
        ["split", "Sample", "agenum", "ExperimentalCondition"]
    ]
    chromage_grouped = boot_chromage.groupby(["split", "ExperimentalCondition"])
    # Get train accuracy
    yo_score = pd.concat(
        [
            chromage_grouped.get_group(("train", "Young")),
            chromage_grouped.get_group(("train", "Old")),
        ]
    )
    yo_pred = yo_score["ChromAge"] > 0.5
    yo_true = yo_score["ExperimentalCondition"] == "Old"
    boot_train_accuracy = accuracy_score(yo_true, yo_pred)
    boot_train_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    yo_score = pd.concat(
        [
            chromage_grouped.get_group(("test", "Young")),
            chromage_grouped.get_group(("test", "Old")),
        ]
    )
    yo_pred = yo_score["ChromAge"] > 0.5
    yo_true = yo_score["ExperimentalCondition"] == "Old"
    boot_test_accuracy = accuracy_score(yo_true, yo_pred)
    boot_test_confusion = pd.Series(
        data=confusion_matrix(yo_true, yo_pred).ravel(),
        index=["true_neg", "false_pos", "false_neg", "true_pos"],
    )

    return (
        boot_chromage,
        boot_train_accuracy,
        boot_train_confusion,
        boot_test_accuracy,
        boot_test_confusion,
        boot_accuracy_curve,
        group_sizes,
    )
