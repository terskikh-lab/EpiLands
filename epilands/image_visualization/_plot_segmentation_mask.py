from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from stardist.plot import render_label
from typing import Tuple

from ..generic_read_write import save_matplotlib_figure


def plot_segmentation_mask(
    image: np.array,
    masks: np.array,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Plots the segmentation mask of an input image.

    Args:
        image (np.array): The input image to be segmented.
        masks (np.array): The segmentation mask of the input image.

    Returns:
        Tuple[Figure, Tuple[Axes, Axes]]: A tuple containing the matplotlib figure and the axes of the plotted images.
    """
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(4, 8),
        # dpi=DEFAULT_DPI
    )
    axs[0].imshow(image, cmap="gray")
    axs[0].set_axis_off()
    axs[0].set_title("input image")
    masked_img = render_label(masks, img=image)
    axs[1].imshow(masked_img, cmap="gray")
    axs[1].set_axis_off()
    axs[1].set_title("Segmentation + input overlay")
    return fig, (ax1, ax2)
