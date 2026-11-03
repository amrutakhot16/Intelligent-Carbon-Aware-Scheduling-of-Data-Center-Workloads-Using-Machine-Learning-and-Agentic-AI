"""
Run the whole pipeline end to end.

    python code/main.py                 # fast mode (small searches / fewer epochs, minutes on a laptop)
    python code/main.py --full          # bigger searches, full-length deep-learning training
    python code/main.py --skip-eda      # skip the EDA plots
    python code/main.py --skip-rf       # skip Random Forest (slowest classic model)
    python code/main.py --skip-dl       # skip LSTM, GRU and TFT (no PyTorch needed)
    python code/main.py --skip-tft      # train LSTM and GRU only (TFT is the slowest deep model)

Outputs: data/processed/model_table.parquet, data/models/*, resources/figures/*.png, resources/*.csv
"""
import argparse
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")          # save figures to files, never open windows

import joblib
import pandas as pd

import settings
from settings import CAT_FEATURES, HORIZON, MODEL_DIR, NUM_FEATURES, PROC_DIR, TARGET
from data_loading import download_data, load_all
from eda import plot_feature_correlation, run_eda
from evaluation import comparison_table, evaluate, per_region_mae, plot_forecast_examples, plot_model_comparison
from features import build_model_table, chronological_split, get_xy
from models import linear_regression, random_forest, xgboost_model
from models.baselines import predict_baselines
from preprocessing import preprocess
from scheduling import build_cubes, example_recommendation, spatio_temporal_savings, time_shift_savings

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160, "display.max_columns", 40)

ALL_MODELS = ["Linear Regression", "Random Forest", "XGBoost", "LSTM", "GRU", "TFT"]


def section(title: str) -> None:
    """
    Print a header line so the console output is easy to follow.

    :param title: section title
    :return: None
    """
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")