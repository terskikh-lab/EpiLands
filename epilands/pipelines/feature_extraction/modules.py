from __future__ import annotations
import os
import logging
import time
import copy
import numpy as np
import pandas as pd
from beartype import beartype
from tqdm import tqdm, trange
import basicpy
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Union, Optional, Callable, List, Tuple, Dict, Any, Union, Optional
from functools import partial

from ...image_segmentation import (
    segment_image_stardist2d,
    segment_image_stardist3d,
)


from multiprocesspipelines.module import Module
from ...image_qc import (
    check_image_shapes,
    power_spectrum_loglog_slope,
    percent_max,
    percent_median2SD,
    threshold_series,
)
from ...tools import (
    join_tuple,
    reorder_iterable,
)
from ...image_visualization import (
    plot_image_data_dict,
)
from ...read_write import (
    save_dataframe_to_csv,
    save_dataframe_to_h5_file,
    extract_imagelike_file_information,
    read_segmentation_data,
)
from ...config import ALL_SUBDIRS

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


class ObjectInput(Module):
    def __init__(
        self,
        name: str,
        object_image_directory: str,
        output_directory: str,
        search_pattern: Union[str, re.Pattern] = ".hdf5",
        metadata_pattern: Union[str, re.Pattern] = re.compile(
            "row([0-9]+)col([0-9]+)fov([0-9]+)"
        ),
        channelIndex: int = None,
        rowIndex: int = 0,
        colIndex: int = 1,
        zIndex: int = None,
        FOVIndex: int = 2,
        tIndex: int = None,
    ):
        super().__init__(
            name=name,
            output_directory=output_directory,
            loggers=["MultiProcessTools", *ALL_SUBDIRS],
        )
        if not os.path.isdir(object_image_directory):
            self.create_directory(object_image_directory)
            object_image_directory = self.get_directory(object_image_directory)
        self.file_information = extract_imagelike_file_information(
            file_path=object_image_directory,
            search_pattern=search_pattern,
            metadata_pattern=metadata_pattern,
            channelIndex=channelIndex,
            rowIndex=rowIndex,
            colIndex=colIndex,
            zIndex=zIndex,
            FOVIndex=FOVIndex,
            tIndex=tIndex,
        )
        self.file_information_grouped = self.file_information.sample(
            frac=1, replace=False
        ).groupby(["row", "column", "FOV"])

    def _load_segmentation_result(self, segmentation_filepath):
        (
            self.objects,
            self.details,
        ) = read_segmentation_data(segmentation_filepath)


class ObjectFeatureTransformer(ObjectInput):
    def _create_feature_dataframe(self, wellIdx, field_of_view):
        # create empty dataframe to append data to, indexed by cell number
        # add in the relevant data for wellindex, FOV, x&y coordinates, object area, morphology
        df_feature_data = pd.DataFrame(index=self.objects.keys())
        df_feature_data["WellIndex"] = int(wellIdx)
        df_feature_data["FieldOfView"] = int(field_of_view)
        if self.details["points"].shape[1] == 2:
            df_feature_data["XCoord"] = self.details["points"][:, 0]
            df_feature_data["YCoord"] = self.details["points"][:, 1]
        elif self.details["points"].shape[1] == 3:
            df_feature_data["ZCoord"] = self.details["points"][:, 0]
            df_feature_data["XCoord"] = self.details["points"][:, 1]
            df_feature_data["YCoord"] = self.details["points"][:, 2]
        else:
            raise NotImplementedError(
                f"details['points'].shape is not 2D or 3D but \
                {self.details['points'].shape[1]}D, what is the dimension of your images??"
            )
        df_feature_data["STARDIST_probability"] = self.details["prob"]
        self.df_feature_data = df_feature_data

    def _append_feature_data(self):
        # add the TAS data to the dataframe
        self.df_feature_data = self.df_feature_data.merge(
            pd.DataFrame.from_dict(self.features, orient="index"),
            left_index=True,
            right_index=True,
        )

    def threshold_object_size(self, name, size_thresh_low, size_thresh_high):
        logger.info(f"{len(self.objects)} objects found in well {name}")
        self.objects, self.outlier_objects = threshold_series(
            self.df_feature_data["MOR_object_pixel_area"],
            size_thresh_low,
            size_thresh_high,
        )
        logger.info(
            f"Object size profile:\n{self.df_feature_data['MOR_object_pixel_area'].describe()}"
        )
        logger.info(f"{len(self.objects)} objects left after size thresholding")

    def run(self):
        logger.info(f"Running Feature Extraction")
        # initialize run information
        # segmentation_output_directory = self.directories['segmentation_output_directory']
        try:
            self.create_directory("feature_extraction")
            output_directory = self.get_directory("feature_extraction")
            for row_col_fov, group_data in tqdm(
                self.file_information_grouped, "Progress extracting features:"
            ):
                # Create a new name to save the segmenetation results for this set of images
                final_file_name = (
                    group_data["filename"].iloc[0].replace("_segmented", "_features")
                )
                temp_file_name = final_file_name.replace("hdf5", "tmp")
                file_not_in_use = self.create_temp_file(
                    final_file_name=final_file_name,
                    temp_file_name=temp_file_name,
                    path="feature_extraction",
                )
                if file_not_in_use == False:
                    logger.debug(f"File {temp_file_name} already exists, skipping...")
                    continue
                group_name = (
                    f"row{row_col_fov[0]}col{row_col_fov[1]}fov{row_col_fov[2]}"
                )
                # extract wellindex and FOV, check if it matches segmentation images. Raise error if not
                logger.info(f"Analyzing data from image: {group_name}....")

                self._load_segmentation_result(group_data["file_path"].iloc[0])

                if len(self.objects) == 0:
                    logger.warning(f"Image {group_name} had no segmented objects")
                    continue

                self._create_feature_dataframe(
                    wellIdx=group_data["WellIndex"].iloc[0],
                    field_of_view=group_data["FOV"].iloc[0],
                )

                self.run_all_processes()

                self._append_feature_data()
                # np.seterr(invalid="warn")
                save_dataframe_to_h5_file(
                    os.path.join(output_directory, final_file_name),
                    self.df_feature_data,
                )
                self.delete_tempfile(
                    os.path.join(
                        self.get_directory("feature_extraction"), temp_file_name
                    )
                )
            self.cleanup()
        except Exception as e:
            logger.exception(e)
            logger.error("An exception occurred, cleaning up...")
            self.cleanup()

    def run_multiprocessing(self, process):
        logger.info(f"Running Feature Extraction")
        # initialize run information
        # segmentation_output_directory = self.directories['segmentation_output_directory']
        try:
            self.create_directory("feature_extraction")

            def _run_multiprocessing(group_data):
                final_file_name = (
                    group_data["filename"].iloc[0].replace("_segmented", "_features")
                )
                temp_file_name = final_file_name.replace("hdf5", "tmp")
                file_not_in_use = self.create_temp_file(
                    final_file_name=final_file_name,
                    temp_file_name=temp_file_name,
                    path="feature_extraction",
                )
                if file_not_in_use == False:
                    logger.debug(f"File {temp_file_name} already exists, skipping...")
                    return
                process(
                    output_directory=self.get_directory("feature_extraction"),
                    group_data=group_data,
                )
                self.delete_tempfile(
                    os.path.join(
                        self.get_directory("feature_extraction"), temp_file_name
                    )
                )

            with ProcessPoolExecutor() as executor:
                # Create a new name to save the segmenetation results for this set of images
                futures = executor.map(
                    _run_multiprocessing,
                    [
                        group_data
                        for row_col_fov, group_data in self.file_information_grouped
                    ],
                )
                for future in tqdm(
                    as_completed(futures), "Progress extracting features:"
                ):
                    pass
                executor.shutdown(wait=True)
            self.cleanup()
        except Exception as e:
            logger.exception(e)
            logger.error("An exception occurred, cleaning up...")
            self.cleanup()


# MERGING ELTA IN WITH ELTAIMAGE


class FeatureModelTransformer(Module):
    def __init__(
        self,
        name: str,
        output_directory: str,
        loggers: List[str],
    ):
        super().__init__(
            name=name,
            output_directory=output_directory,
            loggers=loggers,
        )

    @property
    def channels(self):
        return (
            self.features.columns.str.extract(r"([a-zA-Z0-9]+)_TXT")[0]
            .dropna()
            .unique()
        )

    def get_feature_cols(self, pattern: str, regex: bool = False):
        return list(
            self.features.columns[
                self.features.columns.str.contains(pattern, regex=regex)
            ]
        )

    def get_observation_cols(self, pattern: str, regex: bool = False):
        return list(
            self.observations.columns[
                self.observations.columns.str.contains(pattern, regex=regex)
            ]
        )

    def __getitem__(self, key):
        raise NotImplementedError("Haven't implemented slicing yet")
        self_copy = copy.deepcopy(self)
        for attr in dir(self_copy):
            if attr.startswith("__"):
                continue
            attrval = self_copy.__getattribute__(attr)
            if isinstance(attrval, pd.DataFrame):
                self_copy.__setattr__(attr, attrval.__getitem__(key))
        # self_copy.features = self.features.__getitem__(key)
        # self_copy.observations = self.observations.__getitem__(key)
        return self_copy


# class FeatureModelConstructor(ObjectFeatureInput):

#     def

# class ObjectFeatureVisualization(ObjectInput):

#     def cellcounts(self):

#         # object count for segmentation QA
#         self.df_imageQA.loc[row_col_fov, "object_count"] = len(self.objects)
#         counts_description = pd.Series(self.counts[1:]).describe().to_dict()
#         for metric in counts_description:
#             if metric != "count":
#                 self.df_imageQA.loc[
#                     row_col_fov, f"{metric}_object_size"
#                 ] = counts_description[metric]


#     def run(self):
#         try:
#             def whole_process(objects: Dict[str, np.ndarray]) -> List[pd.Series]:
#                 objects = objects.copy()
#                 image_features = []
#                 for process in self._processes:
#                     result = process(objects)
#                     if isinstance(result, dict):
#                         objects = result
#                     if isinstance(result, pd.Series):
#                         image_features.append(result)
#                 return image_features

#             self.create_directory("object_quality_assessment")
#             output_directory = self.get_directory("object_quality_assessment")
#             self.segmentation_files = []
#             # iterate through every file in the given file dict/segmentation channel pair to segment each set of images
#             for row_col_fov, group_file_information in tqdm(
#                 self.image_file_information_grouped,
#                 f"Progress assessing images: ",
#             ):
#                 final_file_name = f"row{row_col_fov[0]}col{row_col_fov[1]}fov{row_col_fov[2]}_imageQA.csv"
#                 temp_file_name = final_file_name.replace(".csv", ".tmp")
#                 file_not_in_use = self.create_temp_file(
#                     final_file_name=final_file_name,
#                     temp_file_name=temp_file_name,
#                     path=output_directory,
#                 )
#                 if file_not_in_use == False:
#                     logger.info(f"File {temp_file_name} already exists, skipping...")
#                     return file_not_in_use
#                 logger.info(f"Running image QA for row{row_col_fov[0]}col{row_col_fov[1]}fov{row_col_fov[2]}")
#                 wellidx = group_file_information.iloc[0]["row", "column", "FOV", "WellIndex"]
#                 self._load_group_image_data(
#                     group_file_information=group_file_information
#                 )
#                 feature_list = whole_process(self.objects)
#                 feature_list.append(wellidx)
#                 df_imageQA = pd.concat(feature_list).T
#                 logger.info(f"Saving image QA data to {output_directory}")
#                 save_dataframe_to_csv(
#                     df=df_imageQA, path=output_directory, filename=final_file_name
#                 )
#                 self.delete_tempfile(os.path.join(output_directory, temp_file_name))
#                 logger.info(f"Finished image QA for row{row_col_fov[0]}col{row_col_fov[1]}fov{row_col_fov[2]}")
#         except Exception as e:
#             logger.error("An exception occurred, cleaning up...")
#             logger.error(e)
#             self.cleanup()

# if __name__ == "__main__":
