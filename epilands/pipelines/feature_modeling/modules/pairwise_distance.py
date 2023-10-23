import os
import pandas as pd
import epilands as el
import matplotlib.pyplot as plt
from workspace.feature_modeling.processes.zscore_features import zscore_data
from ...create_script_output_dirs import create_script_output_dirs


def MIEL72_zscored_pairwise_distance(
    loadPathRoot: str,
    file: str,
    savePathRoot: str,
    observations: list,
    subset: str,
    metric: str = "euclidean",
):
    outputDir = create_script_output_dirs(
        savePathRoot=savePathRoot, script_name="pairwise_distance"
    )

    data = pd.read_csv(os.path.join(loadPathRoot, file))

    data = zscore_data(data=data.fillna(0), group_by="Tissue", subset=subset)

    subset = data.columns[
        data.columns.str.contains(subset) & ~data.columns.str.contains("CD3")
    ]
    dm = el.feature_preprocessing.distance_matrix_pdist(
        data=data[subset], metric=metric
    )

    dm = pd.DataFrame(data=dm, index=data[observations], columns=data[observations])

    el.read_write.save_dataframe_to_csv(
        df=dm, path=outputDir, filename=f"{metric}_{subset}_distance_matrix"
    )

    return outputDir
