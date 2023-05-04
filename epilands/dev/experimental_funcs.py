import os
from stardist.models import StarDist2D
import pandas as pd
import numpy as np

from .qc import threshold_series
from .preprocessing import glaylconvert
from .read_write import find_all_files, read_all_h5_outputs, convert_dataframe_to_h5ad

MODEL_2D_FLUORESCENT_CELL_SEGMENTATION = StarDist2D.from_pretrained("2D_versatile_fluo")


def segment_single_tiff(tmpImg):
    # obtain an image in the segmentation channel, normalize, generate dataset names
    normImg = glaylconvert(
        tmpImg, np.percentile(tmpImg, 1), np.percentile(tmpImg, 99), 0, 1
    )
    # segment the image, save masks/details to the dictionary
    masks, details = MODEL_2D_FLUORESCENT_CELL_SEGMENTATION.predict_instances(normImg)
    return masks, details


def get_thresholded_mask(masks, size_thresh_low, size_thresh_high, **kwargs):
    masks_in = masks.copy()
    masks_out = masks.copy()
    colors, counts = np.unique(masks.reshape(-1, 1), return_counts=True, axis=0)
    colors = list(np.delete(colors, 0))

    size_series = pd.Series(colors).map(
        lambda i: np.count_nonzero(np.where(masks == i, 1, 0))
    )
    in_cells = threshold_series(size_series, size_thresh_low, size_thresh_high)

    for i in size_series.index:
        if i not in in_cells:
            masks_in = np.where(masks_in == i, 0, masks_in)
        if i in in_cells:
            masks_out = np.where(masks_out == i, 0, masks_out)
    return masks_in, masks_out, size_series


def aggregate_feature_extraction_outputs_to_anndata(
    mielname,
    channels,
    segmentation_output_directory,
    feature_extraction_output_directory,
):
    # find all files matching the pattern for the platemap
    platemap_path = find_all_files(
        segmentation_output_directory, search_str="platemap.txt"
    )
    # checks
    if platemap_path is None:
        raise ValueError(
            f"No platemap found in {segmentation_output_directory}, can't continue to extract features"
        )
    elif len(platemap_path) > 1:
        raise ValueError(
            f"More than one platemap found in {segmentation_output_directory}."
            + "\nPlease delete the extra platemap."
        )
    platemap = pd.read_csv(platemap_path[0], sep="\t")

    # read the feature data into a dataframe
    df_features = read_all_h5_outputs(
        mielname=mielname,
        platemap=platemap,
        file_folder_loc=feature_extraction_output_directory,
        raw_data_pattern="_features",
    )

    # Structure Metadata
    # Here we are going to start structuring some of the metadata.

    # convert data into anndata object, save as H5AD
    adata = convert_dataframe_to_h5ad(
        df_input=df_features,
        output_directory=os.path.dirname(feature_extraction_output_directory),
        name=mielname,
        channels=channels,
        feature_cols=df_features.columns[df_features.columns.str.contains("TAS")],
        observation_cols=df_features.columns[~df_features.columns.str.contains("TAS")],
        reset_index=True,
    )


# import matplotlib.animation as animation
# tmpFileName='r'+('0'+str(rn))[-2:]+'c'+('0'+str(cn))[-2:]+'f'+('0'+str(fn))[-2:]+'p'+('0'+str(pn))[-2:]+'-ch'+str(ch)+'sk1fk1fl1.tiff'
# ani = animation.ArtistAnimation(fig, imgStackShow, interval=50, blit=True,
#                             repeat_delay=1000)
# ani.save(videoSaveFolder+'/r'+('0'+str(rn))[-2:]+'c'+('0'+str(cn))[-2:]+'f'+('0'+str(fn))[-2:]+'-ch'+str(ch)+'sk1fk1fl1.mp4')
# tiff.imsave(projSaveFolder+'/r'+('0'+str(rn))[-2:]+'c'+('0'+str(cn))[-2:]+'f'+('0'+str(fn))[-2:]+'p01-ch'+str(ch)+'sk1fk1fl1.tiff', slicedImg.max(axis=2))
