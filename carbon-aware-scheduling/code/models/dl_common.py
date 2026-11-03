"""
Shared code for the deep-learning models: LSTM, GRU and TFT.

A sample is still one row of the modelling table (region r, target hour T, forecast made 24 h earlier), exactly as
for Linear Regression / Random Forest / XGBoost, so every model is scored on the same 2024 rows.
The difference is the input: instead of ONE feature row, the networks see the last `seq_len` feature rows of the
same region.
"""

import copy
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from settings import DL, FIG_DIR, MODEL_DIR, NUM_FEATURES, SEED, TARGET, VAL_START

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqData:
    """
    Holds the train and test rows sorted by region and time, with the features scaled once.
    The input windows are not stored; they are built batch by batch when needed, which
    keeps the memory use low.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, seq_len: int = DL["seq_len"]):
        """
        :param train: training rows from chronological_split()
        :param test: test rows from chronological_split()
        :param seq_len: number of past hours in each input window
        """
        self.seq_len, self.n_train = seq_len, len(train)
        full = pd.concat([train, test], ignore_index=True)
        full["_pos"] = np.arange(len(full))                      # position in the original train+test order
        full = full.sort_values(["region", "ts"]).reset_index(drop=True)

        gaps = int((full.groupby("region").ts.diff().dropna() != pd.Timedelta(hours=1)).sum())
        print(f"Sequence data: {len(full):,} rows | window = {seq_len} h | time gaps inside regions: {gaps}")

        self.regions = sorted(full.region.unique())
        self.n_regions, self.n_features = len(self.regions), len(NUM_FEATURES)
        self.pos = full["_pos"].to_numpy()
        is_train = self.pos < self.n_train
        self.is_val = is_train & (full.ts >= VAL_START).to_numpy()
        self.is_fit = is_train & ~self.is_val

        # the scaler sees TRAINING rows only
        self.scaler = StandardScaler().fit(full.loc[is_train, NUM_FEATURES])
        self.X = torch.from_numpy(self.scaler.transform(full[NUM_FEATURES]).astype(np.float32))
        region_to_id = {r: i for i, r in enumerate(self.regions)}
        self.region_id = torch.from_numpy(full.region.map(region_to_id).to_numpy().astype(np.int64))
        self.block_start = (np.arange(len(full)) - full.groupby("region").cumcount().to_numpy()).astype(np.int64)

        self.y = full[TARGET].to_numpy(dtype=np.float32)
        self.base = full["ci_lag24"].to_numpy(dtype=np.float32)
        self.scale = float((self.y - self.base)[is_train].std())      # size of a typical 24-h change
        self.target_t = torch.from_numpy((self.y - self.base) / self.scale)
        self._offsets = np.arange(-seq_len + 1, 1)

    def batch(self, rows: np.ndarray):
        """
        Build the input windows for a batch of rows.

        :param rows: positions in the sorted arrays
        :return: (x of shape [batch, seq_len, n_features], region ids [batch], scaled targets [batch])
        """
        idx = rows[:, None] + self._offsets[None, :]
        idx = np.maximum(idx, self.block_start[rows][:, None])        # never reach into another region
        r = torch.from_numpy(rows)
        return self.X[torch.from_numpy(idx)].to(DEVICE), self.region_id[r].to(DEVICE), self.target_t[r].to(DEVICE)

def build_sequence_data(train: pd.DataFrame, test: pd.DataFrame) -> SeqData:
    """
    Create the sequence data used by all three networks.

    :param train: training rows
    :param test: test rows
    :return: SeqData object
    """
    return SeqData(train, test)


@torch.no_grad()
def predict_rows(model: nn.Module, data: SeqData, rows: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    """
    Predict carbon intensity (gCO₂/kWh) for the given rows.

    :param model: trained network
    :param data: SeqData object
    :param rows: positions in the sorted arrays
    :param batch_size: rows per forward pass
    :return: predictions in gCO₂/kWh
    """
    model.eval()
    out = np.concatenate([model(*data.batch(rows[i:i + batch_size])[:2]).cpu().numpy()
                          for i in range(0, len(rows), batch_size)])
    return data.base[rows] + out * data.scale
