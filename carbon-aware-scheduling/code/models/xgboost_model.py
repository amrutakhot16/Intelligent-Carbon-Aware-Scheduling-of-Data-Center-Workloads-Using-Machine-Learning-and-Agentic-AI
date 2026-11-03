"""
Step 10: Model 3 — XGBoost (gradient-boosted trees).

1. RandomizedSearchCV + TimeSeriesSplit on a subsample for the tree hyper-parameters.
2. Early stopping on a time-based validation set (Oct–Dec 2023) to choose the number of trees.
3. Refit on ALL training data with that number of trees.
"""
import time

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from settings import FIG_DIR, SEED, VAL_START
from features import make_preprocessor

XGB_SPACE = {"max_depth": [4, 6, 8, 10], "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10],
             "reg_lambda": [0.5, 1, 5], "reg_alpha": [0, 0.5, 2]}


def train_xgboost(train: pd.DataFrame, X_train, y_train, X_test, Xs, ys, fast_mode: bool = True):
    """
    Tune, early-stop and train the final XGBoost model.

    :param train: training table (its ts column is used to find the validation months)
    :param X_train: training features
    :param y_train: training target
    :param X_test: test features
    :param Xs: features of the time-ordered training subsample used for tuning
    :param ys: target of the tuning subsample
    :param fast_mode: try 10 parameter combinations instead of 40
    :return: (preprocessor, final model, train predictions, test predictions, info for the plots)
    """
    prep = make_preprocessor(scale=False).fit(X_train)
    Xtr_m, Xte_m, Xs_m = prep.transform(X_train), prep.transform(X_test), prep.transform(Xs)

    # 1. hyper-parameter search
    t0 = time.time()
    search = RandomizedSearchCV(xgb.XGBRegressor(n_estimators=400, tree_method="hist", random_state=SEED, n_jobs=-1),
                                XGB_SPACE, n_iter=10 if fast_mode else 40, cv=TimeSeriesSplit(n_splits=3),
                                scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=1)
    search.fit(Xs_m, ys)
    print(f"XGB search time {time.time() - t0:.0f}s | best CV MAE {-search.best_score_:.2f} | best {search.best_params_}")

    # 2. early stopping on the last 3 months of training
    val = (train.ts >= VAL_START).values
    t0 = time.time()
    es_model = xgb.XGBRegressor(n_estimators=3000, tree_method="hist", random_state=SEED, n_jobs=-1,
                                early_stopping_rounds=100, eval_metric="mae", **search.best_params_)
    es_model.fit(Xtr_m[~val], y_train[~val],
                 eval_set=[(Xtr_m[~val], y_train[~val]), (Xtr_m[val], y_train[val])], verbose=False)
    best_iter = es_model.best_iteration
    print(f"Early stopping picked {best_iter} trees")

    # 3. refit on all training data
    final = xgb.XGBRegressor(n_estimators=best_iter + 1, tree_method="hist", random_state=SEED, n_jobs=-1,
                             **search.best_params_)
    final.fit(Xtr_m, y_train)
    print(f"XGBoost fit time {time.time() - t0:.0f}s")

    info = {"best_iter": best_iter, "evals": es_model.evals_result(), "feature_names": prep.get_feature_names_out()}
    return prep, final, final.predict(Xtr_m), final.predict(Xte_m), info


def plot_curve_and_importance(final: xgb.XGBRegressor, info: dict, top_n: int = 15, fig_dir=FIG_DIR) -> None:
    """
    Left: train and validation MAE for each boosting round, with the chosen number of trees
    marked (used to check for overfitting). Right: the most important features by gain.

    :param final: final trained model
    :param info: info dict returned by train_xgboost()
    :param top_n: number of features to show
    :param fig_dir: folder to save the figure in
    :return: None
    """
    ev, feat_names = info["evals"], info["feature_names"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(ev["validation_0"]["mae"], label="train")
    axes[0].plot(ev["validation_1"]["mae"], label="validation (Oct–Dec 2023)")
    axes[0].axvline(info["best_iter"], ls="--", c="grey")
    axes[0].set(title="XGBoost learning curve (overfitting check)", xlabel="boosting round", ylabel="MAE")
    axes[0].legend()
    gain = pd.Series(final.get_booster().get_score(importance_type="gain"))
    gain.index = [feat_names[int(k[1:])] for k in gain.index]
    gain.sort_values()[-top_n:].plot.barh(ax=axes[1], color="#2b6f8f")
    axes[1].set(title=f"XGBoost — top {top_n} features (gain)")
    plt.tight_layout(); plt.savefig(fig_dir / "xgb_curve_importance.png", dpi=120); plt.close()
