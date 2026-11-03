"""
Step 9: Model 2 — Random Forest (bagged tree ensemble).

Hyper-parameters are tuned with RandomizedSearchCV + TimeSeriesSplit on a time-ordered subsample of the
training set; the final model is then fitted on all training data.
"""
import time

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from settings import FIG_DIR, SEED
from features import make_preprocessor

RF_SPACE = {"model__max_depth": [12, 18, 24, None],
            "model__min_samples_leaf": [1, 3, 5, 10],
            "model__max_features": [0.3, 0.5, 0.7, 1.0]}


def tune_random_forest(Xs, ys, fast_mode: bool = True) -> dict:
    """
    Randomized search over RF_SPACE with time-series cross-validation. A smaller forest
    (80 trees) is used during the search to keep it fast.

    :param Xs: features of the time-ordered training subsample
    :param ys: target of the time-ordered training subsample
    :param fast_mode: try 6 combinations instead of 20
    :return: best parameters, without the "model__" prefix
    """
    pipe = Pipeline([("prep", make_preprocessor(scale=False)),
                     ("model", RandomForestRegressor(n_estimators=80, n_jobs=-1, random_state=SEED))])
    t0 = time.time()
    search = RandomizedSearchCV(pipe, RF_SPACE, n_iter=6 if fast_mode else 20, cv=TimeSeriesSplit(n_splits=3),
                                scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=1)
    search.fit(Xs, ys)
    print(f"RF search time {time.time() - t0:.0f}s | best CV MAE {-search.best_score_:.2f} | best {search.best_params_}")
    cv_table = (pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
                .assign(mean_test_MAE=lambda d: -d.mean_test_score).sort_values("mean_test_MAE"))
    print(cv_table[["params", "mean_test_MAE", "std_test_score"]].to_string())
    return {k.replace("model__", ""): v for k, v in search.best_params_.items()}


def train_random_forest(X_train, y_train, X_test, best_params: dict):
    """
    Train the final forest on the full training set with the tuned parameters.
    Each tree sees a random 30% of the rows, which keeps training time and memory manageable.

    :param X_train: training features
    :param y_train: training target
    :param X_test: test features
    :param best_params: output of tune_random_forest()
    :return: (fitted pipeline, train predictions, test predictions)
    """
    t0 = time.time()
    rf = Pipeline([("prep", make_preprocessor(scale=False)),
                   ("model", RandomForestRegressor(n_estimators=200, max_samples=0.3, n_jobs=-1,
                                                   random_state=SEED, **best_params))])
    rf.fit(X_train, y_train)
    print(f"Random Forest final fit {time.time() - t0:.0f}s")
    return rf, rf.predict(X_train), rf.predict(X_test)


def plot_importance(rf: Pipeline, top_n: int = 15, fig_dir=FIG_DIR) -> None:
    """
    Bar chart of the most important features, measured by how much each one reduces the
    error across all splits in the forest (impurity-based importance).

    :param rf: fitted pipeline from train_random_forest()
    :param top_n: number of features to show
    :param fig_dir: folder to save the figure in
    :return: None
    """
    imp = pd.Series(rf.named_steps["model"].feature_importances_,
                    index=rf.named_steps["prep"].get_feature_names_out()).sort_values()[-top_n:]
    ax = imp.plot.barh(figsize=(7, 5), color="#4c8c7a")
    ax.set(title=f"Random Forest — top {top_n} feature importances (impurity)")
    plt.tight_layout(); plt.savefig(fig_dir / "rf_importance.png", dpi=120); plt.close()
