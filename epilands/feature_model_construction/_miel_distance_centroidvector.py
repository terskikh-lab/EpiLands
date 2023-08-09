from __future__ import annotations

# Import libraries
import pandas as pd
import numpy as np
from ..tools import join_iterable
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix

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
        sample_col: str,
        group_col: str,
        feature_cols,
        group_A: str,
        group_B: str,
    ):
        data = data.set_index([sample_col, group_col])
        sample_means = data.groupby([sample_col, group_col])[feature_cols].mean()
        groups = sample_means.groupby(group_col, as_index=True)
        A_centroid = groups.get_group(group_A).mean()
        A_centroid.attrs["name"] = f"{join_iterable(group_A)} centroid"
        B_centroid = groups.get_group(group_B).mean()
        B_centroid.attrs["name"] = f"{join_iterable(group_B)} centroid"

        self.A_centroid = A_centroid
        self.B_centroid = B_centroid
        self.epiAgeVec = B_centroid - A_centroid
        self.l2_norm = np.linalg.norm(self.epiAgeVec.values, ord=2)

        self.coef_ = self.epiAgeVec / self.l2_norm
        self.feature_names_in_ = data.columns
        self.n_features_in_ = len(self.feature_names_in_)
        self.classes_ = [group_A, group_B]

        self.labels = data.index.to_series().apply(lambda idx: group_B in idx)
        self.scores = self.score(data)
        self._roc_auc_analysis(self.scores, self.labels)

    def _scalar_projection(self, data_vec):
        # return np.dot(data_vec, self.epiAgeVec) / self.l2_norm
        return np.dot((data_vec - self.A_centroid), self.epiAgeVec) / self.l2_norm

    def _vector_projection(self, data_vec):
        return np.dot(self._scalar_projection(data_vec), self.epiAgeVec / self.l2_norm)

    def _ortho_projection(self, data_vec):
        return data_vec - self.A_centroid - self._vector_projection(data_vec)

    def _ortho_distance(self, data_vec):
        # return np.linalg.norm(
        #     data_vec - self._vector_projection(data_vec), ord=2
        # )
        return np.linalg.norm(self._ortho_projection(data_vec), ord=2)

    def _roc_auc_analysis(self, scores, labels):
        y_true = labels
        y_score = scores.values.reshape(-1, 1)
        fpr, tpr, thresholds = roc_curve(
            y_true=y_true,
            y_score=y_score,
        )
        auc = roc_auc_score(y_true, y_score)
        if auc < 1:
            threshold = thresholds[np.argmin((tpr - 1) ** 2 + fpr**2)]
        else:
            threshold = (y_score[y_true].min() + y_score[~y_true].max()) / 2
        y_pred = y_score > threshold
        self.auc = auc
        self.threshold = threshold
        self.accuracy_score = accuracy_score(y_true, y_pred)
        self.confusion_matrix = pd.Series(
            data=confusion_matrix(y_true, y_pred).ravel(),
            index=["true_neg", "false_pos", "false_neg", "true_pos"],
        )

    def accuracy_confusion(self, data, labels):
        y_true = labels
        y_pred = self.predict(data)
        accuracy_score = accuracy_score(y_true, y_pred)
        confusion_matrix = pd.Series(
            data=confusion_matrix(y_true, y_pred).ravel(),
            index=["true_neg", "false_pos", "false_neg", "true_pos"],
        )
        return accuracy_score, confusion_matrix

    def score(self, data):
        if any(i not in data.columns for i in self.feature_names_in_):
            raise ValueError("input data missing features")
        EpiAgeDistance = data[self.feature_names_in_].apply(
            self._scalar_projection, axis=1
        )
        EpiAgeDistance.name = "EpiAgeDistance"
        return EpiAgeDistance

    def score_orthogonal(self, data):
        if any(i not in data.columns for i in self.feature_names_in_):
            raise ValueError("input data missing features")
        EpiAgeOrthogonal = data[self.feature_names_in_].apply(
            self._ortho_distance, axis=1
        )
        EpiAgeOrthogonal.name = "EpiAgeOrthogonal"
        return EpiAgeOrthogonal

    def project_orthogonal_subspace(self, data):
        if any(i not in data.columns for i in self.feature_names_in_):
            raise ValueError("input data missing features")
        EpiAgeOrthogonal = data.apply(self._ortho_projection, axis=1)
        # EpiAgeOrthogonal.name = "EpiAgeOrthogonal"
        return EpiAgeOrthogonal

    def predict(self, data):
        y_pred = self.score(data) > self.threshold
        y_pred.name = f"prob_{self.classes_[1]}"
        return y_pred

    def fit_score(
        self,
        data: pd.DataFrame,
        group_by: list,
        group_A: tuple,
        group_B: tuple,
    ):
        self.fit(data, group_by, group_A, group_B)
        return pd.concat([self.scores, EpiAgeOrthogonal], axis=1, join="outer")
