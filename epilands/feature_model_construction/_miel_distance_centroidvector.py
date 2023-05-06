from __future__ import annotations

# Import libraries
import pandas as pd
import numpy as np
from ..tools import join_iterable

# Relative imports


def generate_MIEL_distance_centroidvector(
    df: pd.DataFrame,
    A_centroid: pd.Series,
    B_centroid: pd.Series,
):
    """
    generate_MIEL_distances

    Parameters
    ----------
    df : dataframe
        dataframe of features.

    Returns
    -------
    df_MIEL_distances : dataframe
        pandas dataframe of distances transformed via a linear transformation forming the MIEL/miBioAge axis
    """

    # Calculate centroid for transposition of A to (0,0)
    AB_vec = B_centroid - A_centroid
    AB_mag = np.sqrt(np.sum(np.square(AB_vec.values)))
    # projection of vector A on vetor B is (A . B)/|B|
    func_MIEL_dist = lambda data_vec: np.dot((data_vec - A_centroid), AB_vec) / AB_mag

    def func_MIEL_dist_perp(data_vec):
        data_vec = data_vec - A_centroid
        data_mag = np.sqrt(np.sum(np.square(data_vec)))
        theta = np.arccos(np.dot(data_vec, AB_vec) / (AB_mag * data_mag))
        return data_mag * np.sin(theta)

        # = lambda data_vec: np.sqrt(np.sum(np.square((np.cross(data_vec, AB_vec) / AB_mag))))

    MIEL_distance = df.apply(func_MIEL_dist, axis=1)
    MIEL_orthogonal = df.apply(func_MIEL_dist_perp, axis=1)
    MIEL_distance.name = "MIEL_distance"
    MIEL_orthogonal.name = "MIEL_orthogonal"

    df_MIEL_distance = pd.concat([MIEL_distance, MIEL_orthogonal], axis=1, join="outer")
    return df_MIEL_distance


class EpiAgeModel:
    def __init__(self) -> None:
        pass

    def fit(
        self,
        data: pd.DataFrame,
        group_by: list,
        reference_group_A: tuple,
        reference_group_B: tuple,
    ):
        df_groupby = data.groupby(group_by, as_index=True)
        A_centroid = df_groupby.get_group(reference_group_A).mean()
        A_centroid.attrs["name"] = f"{join_iterable(reference_group_A)} centroid"
        B_centroid = df_groupby.get_group(reference_group_B).mean()
        B_centroid.attrs["name"] = f"{join_iterable(reference_group_B)} centroid"

        self.A_centroid = A_centroid
        self.B_centroid = B_centroid
        self.epiAgeVec = B_centroid - A_centroid
        self.epiAgeVecMag = np.sqrt(np.sum(np.square(self.epiAgeVec.values)))

        self.func_MIEL_dist = (
            lambda data_vec: np.dot((data_vec - A_centroid), self.epiAgeVec)
            / self.epiAgeVecMag
        )

        def func_MIEL_dist_perp(data_vec):
            data_vec = data_vec - A_centroid
            data_mag = np.sqrt(np.sum(np.square(data_vec)))
            theta = np.arccos(
                np.dot(data_vec, self.epiAgeVec) / (self.epiAgeVecMag * data_mag)
            )
            return data_mag * np.sin(theta)

        self.func_MIEL_dist_perp = func_MIEL_dist_perp

    def transform(self, data: pd.DataFrame):
        MIEL_distance = data.apply(self.func_MIEL_dist, axis=1)
        MIEL_orthogonal = data.apply(self.func_MIEL_dist_perp, axis=1)
        MIEL_distance.name = "MIEL_distance"
        MIEL_orthogonal.name = "MIEL_orthogonal"
        df_MIEL_distance = pd.concat(
            [MIEL_distance, MIEL_orthogonal], axis=1, join="outer"
        )
        return df_MIEL_distance

    def fit_transform(
        self,
        data: pd.DataFrame,
        group_by: list,
        reference_group_A: tuple,
        reference_group_B: tuple,
    ):
        self.fit(data, group_by, reference_group_A, reference_group_B)
        return self.transform(data)
