# Import libraries
from __future__ import annotations
import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Callable
from numbers import Number

# from tqdm.notebook import tqdm, trange

# relative imports

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def bootstrap_df(
    df: pd.DataFrame,
    group_by: list,
    with_replacement: bool,
    metric: callable,
    num_bootstraps: int,
    num_cells: int,
    frac: float = None,
    seed: int = None,
):
    """
    Parameters
    ----------
    df : pandas dataframe
        dataframe containing single cell data that has been preprocessed in some way
    groupby : str or list
        column name or list of column names to group by
    metric : callable
        function to apply to each group of cells
    num_bootstraps : int
        number of bootstraps to perform
    num_cells : int
        number of cells to sample per bootstrap
    Returns
    -------
    bootstrap_samples : pandas dataframe
        dataframe containing bootstrap samples
    group_sizes : dict
        dictionary of original group sizes (N)
    seed : int
        random state used in bootstrap sampling
    """
    # logg.info(
    #     f"Began bootstrapping {num_bootstraps} samples of {num_cells} cells from {group_by}"
    # )

    df_groups = df.set_index(group_by, drop=True).groupby(
        group_by
    )  # group by the groupby columns

    if with_replacement == False:
        for grp, _dat in df_groups:
            max_bootstraps = _dat.shape[0] // num_cells
            # num_bootstraps = num_bootstraps * 4 // 3
            if num_bootstraps > max_bootstraps:
                raise ValueError(
                    f"Not enough cells for {num_bootstraps} bootstraps to be in training set for group {grp}"
                )
    if with_replacement == True:
        for grp, _dat in df_groups:
            if _dat.shape[0] < num_cells:
                raise ValueError(f"Not enough cells for bootstrapping in group {grp}")

    bootstrap_samples = []  # create a list to store the bootstrap samples
    if seed == None:
        seed = np.random.randint(low=0, high=2**16, size=None)
    elif not isinstance(seed, Number):
        # logg.error(f"seed {seed} is not of type int")
        raise TypeError(f"seed {seed} is not a number")
    group_sizes = {
        str(group): data.shape[0] for group, data in df_groups
    }  # create a dictionary of group sizes for QC
    # if any(
    #     np.array(list(group_sizes.values())) < num_cells
    # ):  # if any group is too small to sample, raise an error and say why
    #     for group, size in group_sizes.items():
    #         if size < num_cells:
    #             logg.error(
    #                 f"Group {group} has {size} cells, which is less than {num_cells} cells per bootstrap."
    #             )
    #             raise ValueError(
    #                 f"Group {group} has {size} cells, which is less than {num_cells} cells per bootstrap."
    #             )

    # for b in tqdm(range(num_bootstraps)):  # for each bootstrap
    for b in range(num_bootstraps):  # for each bootstrap
        bootstrap_result = df_groups.apply(
            _bootstrap,  # apply the bootstrap function to each group
            num_cells=num_cells,
            frac=frac,
            replace=with_replacement,
            metric=metric,
            seed=seed + b,
        )  # add bootstrap number to seed for initialization
        # bootstrap_result.dropna(
        #     axis=0, how="all", inplace=True  # drop any rows that contain all NaN
        # )
        # if bootstrap_result.index.name in bootstrap_result.columns:
        #     bootstrap_result.drop(
        #         bootstrap_result.index.name, axis="columns", inplace=True
        #     )
        # bootstrap_result.reset_index(inplace=True)  # reset the index
        bootstrap_result["Bootstrap"] = int(b)  # add a column with the bootstrap number
        bootstrap_samples.append(
            bootstrap_result
        )  # add the bootstrap sample to the list
    bootstrap_samples = pd.concat(
        bootstrap_samples
    )  # concatenate the bootstrap samples
    # bootstrap_samples.reset_index(inplace=True, drop=False)
    # logg.info(
    #     f"SUCCESS: bootstrapped {num_bootstraps} samples of {num_cells}"
    #     + f" for {len(df_groups.groups)} groups identified in {group_by}"
    # )
    return bootstrap_samples, group_sizes, seed


def _bootstrap(
    df: pd.DataFrame,
    num_cells: int,
    replace: bool,
    metric: Callable,
    seed: int,
    frac: float = None,
):
    """
    df : pandas dataframe
        dataframe containing a group of cells
    num_cells : int
        number of cells to sample per bootstrap
    metric : callable
        function to apply to each group of cells
    Returns
    -------
    bootstrap_result : pandas dataframe
        dataframe containing the bootstrap sample
    """
    try:
        bootstrap_sample = df.sample(n=num_cells, replace=replace, random_state=seed)
        bootstrap_result = bootstrap_sample.apply(metric, axis=0)
    except ValueError as e:
        if df.shape[0] < num_cells:
            print(f"Not enough cells to bootstrap, returning NaN")
            bootstrap_result = bootstrap_sample.apply(lambda s: np.NaN, axis=0)
        else:
            raise e
    bootstrap_result["original_count"] = int(df.shape[0])
    return bootstrap_result
