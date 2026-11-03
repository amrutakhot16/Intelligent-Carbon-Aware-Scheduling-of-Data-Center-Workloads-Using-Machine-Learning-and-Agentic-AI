"""
Step 11: metrics, model comparison table, per-region errors and comparison plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from settings import FIG_DIR, RESULTS_DIR, TARGET

BASELINE_NAME = "Naive: persistence 24h"


def evaluate(name, y_tr, p_tr, y_te, p_te) -> dict:
    """
    Compute MAE, RMSE and R² on both the train and test sets, plus MAPE on the test set.
    A large gap between train and test error points to overfitting; similar but high
    errors on both point to underfitting.

    :param name: model name, used as the row label in the comparison table
    :param y_tr: true values, train set
    :param p_tr: predictions, train set
    :param y_te: true values, test set
    :param p_te: predictions, test set
    :return: dict with the model name and all metrics
    """
    rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
    mape = lambda a, b: float(np.mean(np.abs((a - b) / np.clip(np.abs(a), 1, None))) * 100)
    return {"model": name,
            "train_MAE": mean_absolute_error(y_tr, p_tr), "test_MAE": mean_absolute_error(y_te, p_te),
            "train_RMSE": rmse(y_tr, p_tr), "test_RMSE": rmse(y_te, p_te),
            "train_R2": r2_score(y_tr, p_tr), "test_R2": r2_score(y_te, p_te),
            "test_MAPE_%": mape(np.asarray(y_te), np.asarray(p_te))}