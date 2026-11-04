"""
Step 8: Model 1 — Linear Regression (traditional baseline).

OLS on standardised features + one-hot region. Fast and interpretable, but only additive linear effects
(it cannot learn that midday solar matters in California but not in Singapore).
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline

from settings import FIG_DIR
from features import make_preprocessor


def train_linear_regression(X_train, y_train, X_test):
    """
    Check the model with time-series cross-validation on the training set, then fit it on
    the full training set. The scaler is inside the pipeline, so in each fold it is fitted
    on that fold's training part only.

    :param X_train: training features
    :param y_train: training target
    :param X_test: test features
    :return: (fitted pipeline, train predictions, test predictions)
    """
    t0 = time.time()
    lr = Pipeline([("prep", make_preprocessor(scale=True)), ("model", LinearRegression())])
    cv = cross_validate(lr, X_train, y_train, cv=TimeSeriesSplit(n_splits=4),
                        scoring="neg_mean_absolute_error", return_train_score=True)
    print("TimeSeriesSplit CV MAE per fold:", np.round(-cv["test_score"], 2),
          "| mean:", round(-cv["test_score"].mean(), 2))
    lr.fit(X_train, y_train)
    print(f"Linear Regression fit time {time.time() - t0:.1f}s")
    return lr, lr.predict(X_train), lr.predict(X_test)


def plot_coefficients(lr: Pipeline, top_n: int = 15, fig_dir=FIG_DIR) -> None:
    """
    Bar chart of the largest coefficients by absolute value (red = increases the forecast,
    green = decreases it). Since the numeric features are standardized, their coefficients
    can be compared with each other.

    :param lr: fitted pipeline from train_linear_regression()
    :param top_n: number of coefficients to show
    :param fig_dir: folder to save the figure in
    :return: None
    """
    coef = pd.Series(lr.named_steps["model"].coef_, index=lr.named_steps["prep"].get_feature_names_out())
    top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:top_n]
    ax = top[::-1].plot.barh(figsize=(7, 5), color=np.where(top[::-1] > 0, "#c0504d", "#4c8c7a"))
    ax.set(title=f"Linear Regression — {top_n} largest standardised coefficients", xlabel="gCO₂/kWh per 1 std")
    plt.tight_layout(); plt.savefig(fig_dir / "lr_coefficients.png", dpi=120); plt.close()
