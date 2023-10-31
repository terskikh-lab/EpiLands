from __future__ import annotations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple

from ..generic_read_write import save_matplotlib_figure


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plots a heatmap of the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to plot.
        title (str): The title of the plot.
        **kwargs: Additional keyword arguments to pass to seaborn.heatmap.

    Returns:
        Tuple[Figure, Axes]: A tuple containing the Figure and Axes objects of the plot.
    """
    fig, ax = plt.subplots(
        figsize=(df.shape[1], df.shape[0]),
    )
    heatmap = sns.heatmap(
        df, ax=ax, square=True, annot=True, annot_kws={"fontsize": 8}, **kwargs
    )
    heatmap.set(title=title)
    plt.tick_params(axis="y", rotation=0)
    return fig, ax
