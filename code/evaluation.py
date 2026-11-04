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


def comparison_table(results: list, save_name: str = "model_results.csv") -> pd.DataFrame:
    """
    Combine the metrics of all models into one table, add the train/test gap and the
    improvement over the persistence baseline, then save it as a CSV.

    :param results: list of dicts returned by evaluate()
    :param save_name: file name of the CSV, saved in the results folder
    :return: comparison table, one row per model
    """
    res = pd.DataFrame(results).set_index("model")
    res["overfit_gap_MAE"] = res["test_MAE"] - res["train_MAE"]
    if BASELINE_NAME in res.index:
        res["vs_persistence_%"] = (1 - res["test_MAE"] / res.loc[BASELINE_NAME, "test_MAE"]) * 100
    res.round(3).to_csv(RESULTS_DIR / save_name)
    print(res.round(3).to_string())
    return res


def plot_model_comparison(res: pd.DataFrame, fig_dir=FIG_DIR) -> None:
    """
    Bar charts of train vs test MAE and of test R² for every model.

    :param res: comparison table from comparison_table()
    :param fig_dir: folder to save the figure in
    :return: None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    res[["train_MAE", "test_MAE"]].plot.bar(ax=axes[0], color=["#9cc5b8", "#2b6f8f"], rot=20)
    axes[0].set(title="MAE (gCO₂/kWh) — lower is better", ylabel="MAE")
    res["test_R2"].plot.bar(ax=axes[1], color="#4c8c7a", rot=20)
    axes[1].set(title="Test R²", ylim=(0.9 * res.test_R2.min(), 1))
    plt.tight_layout(); plt.savefig(fig_dir / "model_comparison.png", dpi=120); plt.close()


def per_region_mae(test: pd.DataFrame, preds: dict, models: list) -> pd.DataFrame:
    """
    Test MAE of each model broken down by region, to see which grids are hardest to forecast.

    :param test: test set (needs the region and target columns)
    :param preds: {model name: test predictions}, in the same row order as test
    :param models: model names to include; the table is sorted by the last one
    :return: table with one row per region and one column per model, plus the region's mean carbon intensity
    """
    out = pd.DataFrame({m: pd.Series(np.abs(test[TARGET].values - preds[m])).groupby(test.region.values).mean()
                        for m in models}).round(1)
    out["region_mean_ci"] = test.groupby("region")[TARGET].mean().round(0)
    out = out.sort_values(models[-1])
    out.to_csv(RESULTS_DIR / "per_region_mae.csv")
    print(out.to_string())
    return out


def plot_forecast_examples(test: pd.DataFrame, preds: dict, regions=("DE-LU", "US-CAL-CISO"),
                           start="2024-07-01", end="2024-07-11", fig_dir=FIG_DIR) -> None:
    """
    Plot actual vs predicted carbon intensity for a few regions over a short test period.
    Germany and California are used by default since they change the most within a day.
    Only the models present in preds are drawn.

    :param test: test set (needs the region, ts and target columns)
    :param preds: {model name: test predictions}, in the same row order as test
    :param regions: regions to plot, one panel each
    :param start: first day to plot (inclusive)
    :param end: last day to plot (exclusive)
    :param fig_dir: folder to save the figure in
    :return: None
    """
    colors = {"Linear Regression": "#c0504d", "Random Forest": "#9b7bb8", "XGBoost": "#2b6f8f",
              "LSTM": "#e08e45", "GRU": "#d4a72c", "TFT": "#3a9d5d"}
    fig, axes = plt.subplots(len(regions), 1, figsize=(14, 3.5 * len(regions)))
    for ax, r in zip(np.atleast_1d(axes), regions):
        m = (test.region == r).values & (test.ts >= start).values & (test.ts < end).values
        ax.plot(test.ts[m], test[TARGET][m], c="black", lw=2, label="actual")
        for name, c in colors.items():
            if name in preds:
                ax.plot(test.ts[m], preds[name][m], lw=1.2, c=c, label=name)
        ax.set(title=f"{r} — 24-h-ahead forecasts (unseen test data)", ylabel="gCO₂/kWh")
        ax.legend(fontsize=8, ncol=4)
    plt.tight_layout(); plt.savefig(fig_dir / "forecast_examples.png", dpi=120); plt.close()
