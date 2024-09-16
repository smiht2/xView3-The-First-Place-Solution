import abc
import os
from abc import abstractmethod
from functools import partial
from typing import Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .base import DatasetSplitter


__all__ = [
    "EvalPublicDatasetSplitter",
]


class EvalPublicDatasetSplitter(DatasetSplitter):
    def __init__(self, data_dir):
        super().__init__(data_dir,'test')
        self.dataset_size = "test"
        self.shore_line = 'public'

    def train_test_split(self, fold: int, num_folds: int):
        # if (fold is not None and fold != 0) or num_folds is not None:
            # raise RuntimeError("Test dataset does not support fold split")

        test_df = pd.read_csv(os.path.join(self.data_dir, "public.csv"))
        test_df["scene_path"] = test_df["scene_id"].apply(partial(self.append_prefix,folder=''))
        test_df["location"] = test_df["scene_id"]
        test_df["folder"] = "test"

        test_path = os.path.join(self.data_dir,'test')
        test_scene_avail = [f.name for f in os.scandir(test_path) if f.is_dir()]
        

        test_df = test_df[test_df.scene_id.isin(test_scene_avail)].reset_index(drop=True)

        # valid_df = valid_df[valid_df.scene_id.isin({"590dd08f71056cacv", "b1844cde847a3942v"})].reset_index(drop=True)

        shore_root = os.path.join(self.data_dir, self.shore_line)
        return test_df,test_df,test_df, shore_root
