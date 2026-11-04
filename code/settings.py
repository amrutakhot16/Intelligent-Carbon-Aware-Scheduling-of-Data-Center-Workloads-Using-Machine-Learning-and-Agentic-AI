"""
Loads the project settings from config.json (edit config.json, not this file).
"""
import json
import os
from pathlib import Path
import pandas as pd

# Repo root = folder containing code/, data/, resources/. Override with PROJECT_ROOT (e.g. on Colab).
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

CONFIG_PATH = ROOT / "config.json"
with CONFIG_PATH.open(encoding="utf-8") as f:
    _cfg = json.load(f)

DATA_DIR = ROOT / _cfg["paths"]["data_dir"]
PROC_DIR = ROOT / _cfg["paths"]["processed_dir"]
MODEL_DIR = ROOT / _cfg["paths"]["models_dir"]
FIG_DIR = ROOT / _cfg["paths"]["figures_dir"]
RESULTS_DIR = ROOT / _cfg["paths"]["results_dir"]
HPE_REPO_URL = _cfg["hpe_repo_url"]

SEED = _cfg["seed"]
HORIZON = _cfg["horizon"]
FAST_MODE = _cfg["fast_mode"]
TRAIN_END = pd.Timestamp(_cfg["train_end"], tz="UTC")
VAL_START = pd.Timestamp(_cfg["val_start"], tz="UTC")

TARGET_RAW = _cfg["target_raw"]
TARGET = _cfg["target"]
CI_LAGS = list(_cfg["ci_lags"])
NUM_FEATURES = [f"ci_lag{L}" for L in CI_LAGS] + list(_cfg["numeric_extra_features"])
CAT_FEATURES = list(_cfg["categorical_features"])

DL = dict(_cfg["deep_learning"])                    # hyper-parameters for the LSTM, GRU and TFT models.

JOB_HOURS = _cfg["scheduling"]["job_hours"]
WINDOW_HOURS = _cfg["scheduling"]["window_hours"]
REFERENCE_REGION = _cfg["scheduling"]["reference_region"]


def make_dirs():
    """
    Create the output folders if they don't exist yet.

    :return: None
    """
    for d in (PROC_DIR, MODEL_DIR, FIG_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
