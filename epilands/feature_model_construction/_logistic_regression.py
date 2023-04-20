# Import libraries
from __future__ import annotations
from typing import Optional, Tuple, Union
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Relative imports
from ..tools import join_iterable


def logistic_regression(
    fit_data: pd.DataFrame,
    transform_data: pd.DataFrame,
    class_cols: list,
    penalty: str = "l1",
    solver: str = "liblinear",
    verbose: int = 1,
    n_jobs: None = 4,
    **kwargs,
) -> Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, str]:
    LR_model = LogisticRegression(
        penalty=penalty, solver=solver, verbose=verbose, n_jobs=n_jobs, **kwargs
    )
    LR_model.fit(
        X=fit_data.loc[:, ~fit_data.columns.isin(class_cols)],
        y=fit_data.loc[:, fit_data.columns.isin(class_cols)],
    )
    data_prob = LR_model.predict_proba(
        X=transform_data.loc[:, ~transform_data.columns.isin(class_cols)]
    )
    uns_data = LR_model.get_params()
    uns_data["labels"] = [join_iterable(["prob", i]) for i in LR_model.classes_]
    return data_prob, uns_data, LR_model
