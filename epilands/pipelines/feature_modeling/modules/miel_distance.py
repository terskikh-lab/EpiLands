import os
import pandas as pd
import epilands as el
import matplotlib.pyplot as plt
from workspace.feature_modeling.processes.zscore_features import zscore_data
from ...create_script_output_dirs import create_script_output_dirs


def MIEL72_miel_distance(
    loadPathRoot: str,
    file: str,
    savePathRoot: str,
    reference_group_A: tuple,
    reference_group_B: tuple,
    group_by: list,
    observations: list,
    subset: str,
    metric: str = "euclidean",
):
    outputDir = create_script_output_dirs(
        savePathRoot=savePathRoot, script_name="miel_distance"
    )

    data = pd.read_csv(os.path.join(loadPathRoot, file))
    data.sample()

    params = {
        "loadPathRoot": loadPathRoot,
        "file": file,
        "subset": subset,
        "reference_group_A": reference_group_A,
        "reference_group_B": reference_group_B,
        "group_by": group_by,
    }

    df_groupby = data.groupby(group_by, as_index=True)
    A_centroid = df_groupby.get_group(reference_group_A).mean()
    A_centroid.attrs["name"] = f"{el.tools.join_iterable(reference_group_A)} centroid"
    B_centroid = df_groupby.get_group(reference_group_B).mean()
    B_centroid.attrs["name"] = f"{el.tools.join_iterable(reference_group_B)} centroid"

    df_MIEL_distance = (
        el.feature_model_construction.generate_MIEL_distance_centroidvector(
            df=df,
            A_centroid=A_centroid,
            B_centroid=B_centroid,
        )
    )
    # display(df_MIEL_distances)
    df_MIEL_distance = self.obs.merge(
        df_MIEL_distance, left_index=True, right_index=True, validate="one_to_one"
    )
    df_MIEL_distance.index = self.obs.index
    self.obsm[name] = df_MIEL_distance[["MIEL_distance", "MIEL_orthogonal"]]
    df_MIEL_distance[self.obs.columns] = self.obs
    df_MIEL_distance.attrs["name"] = name
    if save_info:
        save_dataframe_to_csv(df_MIEL_distance, save_to)

    return outputDir
