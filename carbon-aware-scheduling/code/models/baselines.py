"""
Step 7: naive baselines that every ML model must beat.

* Persistence-24h: tomorrow at this hour = today at this hour (ci_lag24)
* Seasonal-weekly: same hour last week (ci_lag168)
"""
import pandas as pd

# baseline name -> lag column used as its prediction
BASELINES = {"Naive: persistence 24h": "ci_lag24",
             "Naive: same hour last week": "ci_lag168"}


def predict_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Get the baseline predictions, which are just the matching lag columns.

    :param train: training set with the lag feature columns
    :param test: test set with the lag feature columns
    :return: {baseline name: (train predictions, test predictions)}
    """
    return {name: (train[col].values, test[col].values) for name, col in BASELINES.items()}
