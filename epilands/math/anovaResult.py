from __future__ import annotations
# from IPython.display import display

import pandas as pd
import numpy as np

from scipy import stats
from sklearn.metrics import r2_score


def anova_table(aov):
    """
    The function below was created specifically for the one-way ANOVA
    table results returned for Type II sum of squares
    """
    aov["mean_sq"] = aov[:]["sum_sq"] / aov[:]["df"]
    aov["eta_sq"] = aov[:-1]["sum_sq"] / sum(aov["sum_sq"])
    aov["omega_sq"] = (aov[:-1]["sum_sq"] - (aov[:-1]["df"] * aov["mean_sq"][-1])) / (
        sum(aov["sum_sq"]) + aov["mean_sq"][-1]
    )

    cols = ["df", "sum_sq", "mean_sq", "F", "PR(>F)", "eta_sq", "omega_sq"]
    aov = aov[cols]
    return aov


def anova_summary(popt, data, func):
    # https://bookdown.org/egarpor/SSS2-UC3M/simplin-aovfit.html
    # https://reference.wolfram.com/language/tutorial/NumericalOperationsOnData.html

    y_pred = func(data["x"], *popt)

    p1 = 1  # number of constraint for Mean
    p2 = len(popt)  # number of parameters for fitting function, including the constent.

    N_data = len(data["y"])
    y_mean = data["y"].mean()

    SS = (data["y"] ** 2).sum()
    SS_Pred = (y_pred**2).sum()
    MS = SS / p2

    SST = ((y_mean - data["y"]) ** 2).sum()
    SSE = ((y_pred - data["y"]) ** 2).sum()  # Sum squared error
    SSR = ((y_pred - y_mean) ** 2).sum()  # sum of squares due to regression

    anova = pd.DataFrame(
        index=[
            "Model",
            "Treatmnet",
            "Residual Error",
            "Uncorrected Total",
            "Corrected Total",
        ]
    )

    anova.loc[:, "df"] = [p2, p2 - p1, N_data - p2, N_data, N_data - p1]
    anova.loc[:, "sum_sq"] = [SS_Pred, SSR, SSE, SS, SST]
    anova["mean_sq"] = anova.iloc[0:3, :]["sum_sq"] / anova.iloc[0:3, :]["df"]

    anova.loc["Treatmnet", "F"] = (
        anova.loc["Treatmnet", "mean_sq"] / anova.loc["Residual Error", "mean_sq"]
    )
    # ((SST - SSE)/(p2 - p1))/(SSE/(N_data - p2))

    anova.loc["Treatmnet", "PR(>F)"] = 1 - stats.f.cdf(
        anova.loc["Treatmnet", "F"], p2 - p1, 10 - p2, loc=0, scale=1
    )

    aic = AIC(N_data, p2, SSE)
    bic = BIC(N_data, p2, SSE)
    # r2 = r2_score(data['y'], y_pred) # 1 - SSE/SST

    fit_sum = pd.DataFrame(index=["AIC", "BIC", "R-squared", "Adj. R-squared"])
    fit_sum.loc["AIC", ""] = AIC(N_data, p2, SSE)
    fit_sum.loc["BIC", ""] = BIC(N_data, p2, SSE)

    if func == func_lin:
        r2 = r2_score(data["y"], y_pred)
        fit_sum.loc["R-squared", ""] = r2
        fit_sum.loc["Adj. R-squared", ""] = 1 - (1 - r2) * (N_data - 1) / (N_data - p2)

        tables = [
            anova.loc[["Treatmnet", "Residual Error", "Corrected Total"]],
            fit_sum,
        ]

    else:
        r2 = 1 - SSE / SS
        fit_sum.loc["R-squared", ""] = r2
        fit_sum.loc["Adj. R-squared", ""] = 1 - (1 - r2) * N_data / (N_data - p2)

        tables = [
            anova.loc[
                ["Treatmnet", "Residual Error", "Uncorrected Total", "Corrected Total"]
            ],
            fit_sum,
        ]

    # display(tables[0])
    # display(tables[1])

    return tables


def AIC(n, k, RSS):
    """
    AIC - Akaike information criterion

    Equation adopt from:
    https://rip94550.wordpress.com/2010/10/18/regression-1-–-selection-criteria/

    Parameters
    ----------
    n: number of data points
    k: number of fitting parameters
    RSS: residual sum of squares, np.sum(y_pred - y_true)**2
      also known as the sum of squared residuals (SSR)
      or the sum of squared estimate of errors (SSE).

    """
    return n * np.log(2 * np.pi * RSS / n) + 2 * (k + 1) + n


def BIC(n, k, RSS):
    """
    BIC - Bayesian information criterion
    """
    return n * np.log(2 * np.pi * RSS / n) + np.log(n) * (k + 1) + n


def func_exp(x, a, b, c):
    # return a * np.exp(b * x) + c
    return a * np.exp(b * x) + c


def func_inv(x, a, b, c):
    return a / (x - b) + c


def func_lin(x, a, b):
    return (b * x) + a
