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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans

from umap import UMAP

logger = logging.getLogger("modules")


def cluster_kmeans(
    data,
    n_clusters,
    seed,
    subset: str,
    cluster_kwargs: dict,
):
    relevant_features = data.columns[data.columns.str.contains(subset)]
    treated_cells = data["ExperimentalCondition"].str.contains("old_i4F_treated")
    metadata = data.loc[:, ~data.columns.isin(relevant_features)]
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, **cluster_kwargs)
    labels = kmeans.fit_predict(data.loc[treated_cells, relevant_features])
    reducer = UMAP(
        n_components=3,
    )
    projection = reducer.fit_transform(data[relevant_features])
    projection = pd.DataFrame(
        projection, index=data.index, columns=["UMAP1", "UMAP2", "UMAP3"]
    )
    projecionfull = pd.concat([projection, metadata], axis=1)
    projecionfull.loc[treated_cells, "kmeans_cluster"] = labels
    projecionfull.loc[:, "kmeans_cluster"].fillna(3, inplace=True)
    means = projecionfull.groupby(
        ["Sample", "ExperimentalCondition", "kmeans_cluster"]
    ).mean()
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="kmeans_cluster",
        data=means,
    )
    fig.savefig("umap1.png")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="UMAP2",
        y="UMAP3",
        hue="kmeans_cluster",
        data=means,
    )
    fig.savefig("umap2.png")

    fig, ax = plt.subplots()
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="ExperimentalCondition",
        data=means,
    )
    fig.savefig("umap1p.png")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="UMAP2",
        y="UMAP3",
        hue="ExperimentalCondition",
        data=means,
    )
    fig.savefig("umap2p.png")

    pass
