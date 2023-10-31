# Import libraries
from __future__ import annotations
import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple, Union
from numbers import Number
import time

# from tqdm.notebook import tqdm, trange

# relative imports

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def bootstrap_df(
    df: pd.DataFrame,
    group_by: list,
    metric: Callable,
    num_cells: Union[int, str],
    num_bootstraps: int,
    seed: int = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Perform bootstrapping on a pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing single cell data that has been preprocessed in some way.
    groupby : str or list
        Column name or list of column names to group by.
    metric : callable
        Function to apply to each group of cells.
    num_cells : int or str
        Number of cells to sample per bootstrap. If "original", use the original number of cells in each group.
    num_bootstraps : int
        Number of bootstraps to perform.
    seed : int, optional
        Random state used in bootstrap sampling.

    Returns
    -------
    bootstrap_samples : pandas dataframe
        Dataframe containing bootstrap samples.
    group_sizes : pandas series
        Series of original group sizes (N).
    seed : int
        Random state used in bootstrap sampling.
    """

    if seed == None:
        seed = np.random.randint(low=0, high=2**16, size=None)
    elif not isinstance(seed, Number):
        raise TypeError(f"seed {seed} is not a number")
    if isinstance(group_by, str) or len(group_by) == 1:
        df_groups = df.set_index(group_by).drop(group_by, axis=1)
    else:
        df_groups = df.set_index(df[group_by].astype(str).apply("|".join, axis=1)).drop(
            group_by, axis=1
        )
        group_by = "|".join(group_by)
        df_groups.index.name = group_by
    df = None

    group_sizes = df_groups.index.value_counts().sort_index()
    # num_bootstraps = int(
    #     min(group_sizes) // num_cells if num_bootstraps is None else num_bootstraps
    # )
    # if num_bootstraps <= 1:
    #     raise ValueError(f"Not enough cells for bootstrapping")
    if isinstance(num_cells, int):
        if any(group_sizes < num_cells):
            for name, grpsize in group_sizes.items():
                print(name, grpsize)
                if grpsize < num_cells:
                    # logger.warning(
                    #     f"Dropping group, not enough cells for bootstrapping: {grp} ({grpsize}) cells"
                    # )
                    # df_groups.drop(name, inplace=True)
                    logger.error(
                        f"Not enough cells for bootstrapping {num_cells} cells in group: {name} ({grpsize}) cells"
                    )
            raise ValueError(f"Not enough cells for bootstrapping")
        logger.debug(
            f"Began calculating {metric} bootstrapping {num_bootstraps} samples of {num_cells} cells from {group_by}"
        )

    rng = np.random.default_rng(seed=seed)

    if metric == "mean":
        _bootdf = lambda df: df.mean(axis=0)
    elif metric == "std":
        _bootdf = lambda df: df.std(axis=0)
    elif isinstance(metric, Callable):
        _bootdf = lambda df: df.apply(metric, axis=0)

    if num_cells == 1:
        logger.warn(f"num_cells = 1, so returning value rather than {metric}")
        bootstrap_samples = []
        for name, size in group_sizes.items():
            # start = time.perf_counter()
            dfgrp = df_groups.loc[name, :]
            sample = rng.choice(size, size=num_cells * num_bootstraps, replace=True)
            idx = [name] * num_bootstraps
            bootstrap_data = dfgrp.iloc[sample, :]
            bootstrap_data.index = idx
            bootstrap_data.loc["bootstrap"] = 1
            bootstrap_samples.append(
                pd.DataFrame(bootstrap_data, index=[name] * num_bootstraps)
            )
            # stop = time.perf_counter()
            # print(stop - start)

    elif num_cells == "original":
        bootstrap_samples = []
        for name, size in group_sizes.items():
            # start = time.perf_counter()
            dfgrp = df_groups.loc[name, :]
            bootstrap_data = []
            sample = rng.choice(size, size=(num_bootstraps, size), replace=True)
            for i, bsample in enumerate(sample):
                df = _bootdf(dfgrp.iloc[bsample, :])
                df.loc["bootstrap"] = i + 1
                bootstrap_data.append(df)
            bootstrap_samples.append(
                pd.DataFrame(bootstrap_data, index=[name] * num_bootstraps)
            )
            # stop = time.perf_counter()
            # print(stop - start)
    else:
        bootstrap_samples = []
        for name, size in group_sizes.items():
            # start = time.perf_counter()
            dfgrp = df_groups.loc[name, :]
            bootstrap_data = []
            sample = rng.choice(size, size=(num_bootstraps, num_cells), replace=True)
            for i, bsample in enumerate(sample):
                # bsample = sample[b * num_cells : b * num_cells + num_cells]
                df = _bootdf(dfgrp.iloc[bsample, :])
                df.loc["bootstrap"] = i + 1
                bootstrap_data.append(df)
            bootstrap_samples.append(
                pd.DataFrame(bootstrap_data, index=[name] * num_bootstraps)
            )
            # stop = time.perf_counter()
            # print(stop - start)

    bootstrap_samples = pd.concat(
        bootstrap_samples
    )  # concatenate the bootstrap samples

    group_by = group_by.split("|")
    original_cols = bootstrap_samples.index.str.extract(
        "\|".join(["([A-Za-z0-9-_]+)"] * len(group_by))
    ).rename({i: col for i, col in enumerate(group_by)}, axis=1)
    bootstrap_samples.reset_index(inplace=True, drop=True)
    bootstrap_samples[group_by] = original_cols
    bootstrap_samples.set_index(group_by, inplace=True)
    logger.debug(f"Bootstrapping completed")
    return bootstrap_samples, group_sizes, seed
