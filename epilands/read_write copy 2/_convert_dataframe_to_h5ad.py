# import os
# import logging
# import datetime
# import pandas as pd
# from anndata import AnnData

# sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
# logger = logging.getLogger(sub_package_name)


# def convert_dataframe_to_h5ad(
#     df_input: pd.DataFrame,
#     output_directory: str,
#     name: str,
#     channels: list,
#     feature_cols: list,
#     observation_cols: list,
#     reset_index: bool,
#     **kwargs,
# ) -> AnnData:
#     """
#     converts a pandas DataFrame into an anndata annotated dataframe and saves the annotated dataframe and

#     df_input: pandas DataFrame

#     output_directory: str the folder where the features are saved

#     name: str the name of the experiment

#     channels: list of str the channels of the experiment

#     feature_cols: list of str the feature columns of the experiment

#     observation_cols: list of str the observation columns of the experiment

#     reset_index: bool if True the index of the dataframe is reset

#     save: bool if True the info of the experiment is saved

#     **kwargs: dict of additional arguments to be added to the unstructured data
#     """

#     if not os.path.exists(output_directory):
#         logger.error(
#             "Experiment output folder does not exist. Please create the folder and try again."
#         )
#         raise ValueError(
#             "Experiment output folder does not exist. Please create the folder and try again."
#         )
#     elif os.path.exists(os.path.join(output_directory, name + ".h5ad")):
#         logger.error("File already exists. Please delete the file and try again.")
#         raise ValueError("File already exists. Please delete the file and try again.")
#     # create copy of the dataframe
#     df = df_input.copy()

#     if reset_index == True:
#         df.reset_index(inplace=True, drop=True)

#     X = df.loc[:, feature_cols].values  # get the feature matrix
#     obs = df.loc[:, observation_cols]  # get the observation matrix
#     var = df.loc[:, feature_cols].columns.to_frame(
#         name="TAS_Features"
#     )  # get the variable matrix
#     for observation in obs.columns:
#         try:
#             if (obs[observation] == obs[observation].astype(int)).all() == False:
#                 obs[observation] = obs[observation].astype(float)
#             else:
#                 obs[observation] = obs[observation].astype(int)
#         except:
#             try:
#                 obs[observation] = pd.Categorical(obs[observation])
#             except:
#                 logger.error(
#                     f"could not convert {observation} to int, float, or categorical"
#                 )
#                 raise ValueError(
#                     f"could not convert {observation} to int, float, or categorical"
#                 )
#     if len(kwargs.keys()) > 0:
#         uns = kwargs
#         uns["name"] = name
#         uns["channels"] = channels
#         uns["modification_log"] = [f"Created at datetime {datetime.datetime.now()}\n\n"]
#     else:
#         uns = {
#             "name": name,
#             "channels": channels,
#             "modification_log": [f"Created at datetime {datetime.datetime.now()}\n\n"],
#         }

#     adata = AnnData(X=X, obs=obs, var=var, uns=uns)

#     logger.info("Saving data to h5ad file...")
#     logger.info(f"adata.X:\n{adata.X}")
#     logger.info(f"adata.obs:\n{adata.obs}")
#     logger.info(f"adata.var:\n{adata.var}")
#     logger.info(f"adata.uns:\n{adata.uns}")
#     # display(adata.X)
#     # display(adata.obs)
#     # display(adata.var)
#     # display(adata.uns)\
#     adata.write_h5ad(os.path.join(output_directory, name + ".h5ad"))
#     logger.info(f"Success! Saved adata to {output_directory} as {name}.h5ad")
#     return adata
