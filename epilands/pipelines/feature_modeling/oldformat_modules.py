from __future__ import annotations
import os
import logging
import copy
import pandas as pd
import re
from typing import Union, Optional, Callable, List, Tuple, Dict, Any, Union, Optional

from epilands.config import ALL_SUBDIRS
from multiprocesspipelines import Module

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


# MERGING ELTA IN WITH ELTAIMAGE


class FeatureModelTransformer(Module):
    def __init__(
        self,
        name: str,
        output_directory: str,
    ):
        super().__init__(
            name=name,
            output_directory=output_directory,
            loggers=["MultiProcessTools", *ALL_SUBDIRS],
        )

    def get_data(self, pattern: Union[str, re.pattern], regex: bool):
        self.data.columns[self.data.columns.str.contains(pattern, regex=regex)]

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
