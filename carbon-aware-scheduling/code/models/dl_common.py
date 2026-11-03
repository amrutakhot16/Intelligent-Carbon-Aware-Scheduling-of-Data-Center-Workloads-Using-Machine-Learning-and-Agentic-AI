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


def fit(model: nn.Module, data: SeqData, name: str, fast_mode: bool = True):
    """
    Train a network with AdamW and L1 loss (L1 because the main metric is MAE). Training stops
    when the validation MAE (Oct-Dec 2023) has not improved for `patience` epochs, and the
    weights from the best epoch are restored.

    :param model: untrained network
    :param data: SeqData object
    :param name: model name, used in the printed output
    :param fast_mode: fewer epochs and a random subset of the training rows each epoch
    :return: (model, train predictions, test predictions, info), predictions in the original row order
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    max_epochs = DL["max_epochs_fast"] if fast_mode else DL["max_epochs"]
    frac = DL["fast_fraction"] if fast_mode else 1.0              # fast mode: a fresh random subset every epoch
    fit_rows, val_rows = np.where(data.is_fit)[0], np.where(data.is_val)[0]
    n_use, bs = int(len(fit_rows) * frac), DL["batch_size"]
    print(f"{name}: {n_params:,} parameters | device {DEVICE.type} | {n_use:,} samples/epoch | up to {max_epochs} epochs")

    opt = torch.optim.AdamW(model.parameters(), lr=DL["lr"], weight_decay=DL["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = nn.L1Loss()
    history = {"train_MAE": [], "val_MAE": []}
    best, best_state, best_epoch, bad, t_start = np.inf, None, 0, 0, time.time()

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        perm = rng.permutation(fit_rows)[:n_use]
        for i in range(0, len(perm), bs):
            x, reg, y = data.batch(perm[i:i + bs])
            opt.zero_grad()
            loss = loss_fn(model(x, reg), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        tr_mae = float(np.mean(losses)) * data.scale
        val_mae = float(np.mean(np.abs(predict_rows(model, data, val_rows) - data.y[val_rows])))
        sched.step(val_mae)
        history["train_MAE"].append(tr_mae); history["val_MAE"].append(val_mae)
        print(f"  epoch {epoch:2d}  train MAE {tr_mae:6.2f}  val MAE {val_mae:6.2f}  ({time.time() - t0:.0f}s)")
        if val_mae < best - 1e-3:
            best, best_epoch, bad, best_state = val_mae, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= DL["patience"]:
                print(f"  early stopping (best epoch {best_epoch}, val MAE {best:.2f})")
                break
    model.load_state_dict(best_state)
    print(f"{name} fit time {time.time() - t_start:.0f}s | best epoch {best_epoch} | best val MAE {best:.2f}")

    pred_sorted = predict_rows(model, data, np.arange(len(data.y)))
    pred = np.empty(len(pred_sorted), dtype=np.float64)
    pred[data.pos] = pred_sorted                                   # back to the original train + test order
    info = {"history": history, "best_epoch": best_epoch, "n_params": n_params,
            "fit_seconds": time.time() - t_start, "hparams": dict(DL)}
    return model, pred[:data.n_train], pred[data.n_train:], info


def plot_training_curves(infos: dict, fig_dir=FIG_DIR) -> None:
    """
    Plot the training and validation MAE per epoch for each network, with a dot at the
    epoch that was kept.

    :param infos: {model name: info dict returned by fit()}
    :param fig_dir: folder to save the figure in
    :return: None
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (name, info), c in zip(infos.items(), ["#e08e45", "#9b7bb8", "#3a9d5d"]):
        ep = range(1, len(info["history"]["val_MAE"]) + 1)
        ax.plot(ep, info["history"]["val_MAE"], c=c, lw=2, label=f"{name} validation")
        ax.plot(ep, info["history"]["train_MAE"], c=c, lw=1, ls="--", label=f"{name} train")
        ax.scatter([info["best_epoch"]], [info["history"]["val_MAE"][info["best_epoch"] - 1]], c=c, zorder=3)
    ax.set(title="Deep-learning learning curves (dot = chosen epoch)", xlabel="epoch", ylabel="MAE (gCO₂/kWh)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(fig_dir / "dl_learning_curves.png", dpi=120); plt.close()


def save_dl_artifacts(fitted: dict, data: SeqData) -> None:
    """
    Save the weights of each network, plus everything needed to rebuild the input windows
    and undo the scaling when the models are loaded again.

    :param fitted: {model name: (network, info)}
    :param data: SeqData object used for training
    :return: None
    """
    for name, (net, info) in fitted.items():
        torch.save(net.state_dict(), MODEL_DIR / f"{name.lower()}.pt")
    with open(MODEL_DIR / "dl_config.json", "w") as f:
        json.dump({"seq_len": data.seq_len, "NUM_FEATURES": NUM_FEATURES, "regions": data.regions,
                   "scaler_mean": data.scaler.mean_.tolist(), "scaler_scale": data.scaler.scale_.tolist(),
                   "residual_scale": data.scale, "hparams": dict(DL),
                   "models": {n: {"params": i["n_params"], "best_epoch": i["best_epoch"]} for n, (_, i) in fitted.items()}},
                  f, indent=2)
