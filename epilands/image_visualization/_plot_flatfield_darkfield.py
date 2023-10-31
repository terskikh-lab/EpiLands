from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from typing import Tuple, Union

from ..generic_read_write import save_matplotlib_figure


def plot_flatfield_darkfield(
    label: Union[int, str],
    flatfield: np.ndarray,
    darkfield: np.ndarray,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Plots the flatfield and darkfield images side by side.

    Args:
        label (Union[int, str]): The label to be used in the title of the plots.
        flatfield (np.ndarray): The flatfield image to be plotted.
        darkfield (np.ndarray): The darkfield image to be plotted.

    Returns:
        Tuple[Figure, Tuple[Axes, Axes]]: A tuple containing the figure and the axes of the subplots.
    """
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 5),
    )
    axs[0].imshow(flatfield, cmap="gray")
    axs[0].set_title(f"{label} Flatfield")
    axs[1].imshow(darkfield, cmap="gray")
    axs[1].set_title(f"{label} Darkfield")
    return fig, axs
