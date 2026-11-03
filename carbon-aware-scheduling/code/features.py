"""
Step 5-6: feature engineering, chronological train/test split, and the encoding/scaling preprocessor.

Leakage rule: we forecast hour T while standing at T-24h, so every carbon/price feature comes from >= 24 h
before the target hour. Weather at T is used as a stand-in for a weather forecast (a known limitation).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from settings import CAT_FEATURES, CI_LAGS, HORIZON, NUM_FEATURES, TARGET, TRAIN_END


def add_lag_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Add past values of carbon intensity, rolling statistics and lagged grid-mix and price columns.
    All shifts are done per region so values never leak from one region into another.

    :param df: cleaned hourly table with one row per region and hour
    :param horizon: how many hours ahead the forecast is made (24)
    :return: the same table with the lag and rolling feature columns added
    """
    df = df.sort_values(["region", "ts"]).reset_index(drop=True)
    g = df.groupby("region")
    for L in CI_LAGS:
        df[f"ci_lag{L}"] = g["ci"].shift(L)

    s24 = g["ci"].shift(horizon)          # rolling windows END at T-24
    df["ci_roll24_mean"] = s24.groupby(df.region).transform(lambda s: s.rolling(24).mean())
    df["ci_roll24_std"] = s24.groupby(df.region).transform(lambda s: s.rolling(24).std())
    df["ci_roll168_mean"] = s24.groupby(df.region).transform(lambda s: s.rolling(168).mean())
    df["ci_trend24"] = df["ci_lag24"] - df["ci_lag48"]

    for c in ["renewable_pct", "low_carbon_pct", "is_estimated", "price"]:
        df[f"{c}_lag24"] = g[c].shift(horizon)
    return df


def add_calendar_features(df: pd.DataFrame, region_tz: dict) -> pd.DataFrame:
    """
    Add hour of day, day of week and month in each grid's local time, plus a weekend flag.
    Local time matters because demand and solar output follow the local clock, not UTC.
    The values are encoded as sin/cos so that e.g. hour 23 and hour 0 end up close together.

    :param df: table with region and ts (UTC) columns
    :param region_tz: {region: timezone name}
    :return: the same table with the calendar feature columns added
    """
    for r, tz in region_tz.items():
        m = df.region == r
        if not m.any():
            continue
        lt = df.loc[m, "ts"].dt.tz_convert(tz)
        df.loc[m, "hour_local"] = lt.dt.hour.values
        df.loc[m, "dow_local"] = lt.dt.dayofweek.values
        df.loc[m, "month_local"] = lt.dt.month.values
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df.hour_local / 24), np.cos(2 * np.pi * df.hour_local / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * df.dow_local / 7), np.cos(2 * np.pi * df.dow_local / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df.month_local - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df.month_local - 1) / 12)
    df["is_weekend"] = (df.dow_local >= 5).astype(int)
    return df


def build_model_table(df: pd.DataFrame, region_tz: dict) -> pd.DataFrame:
    """
    Build the final modelling table: add all features, then drop the first rows of each
    region, where the lag windows reach back before the start of the data.

    :param df: cleaned and merged hourly table
    :param region_tz: {region: timezone name}
    :return: table with no missing values in the features or the target
    """
    df = add_lag_features(df)
    df = add_calendar_features(df, region_tz)
    n_before = len(df)
    model_df = df.dropna(subset=NUM_FEATURES + [TARGET]).reset_index(drop=True)
    print(f"Dropped {n_before - len(model_df)} rows whose lag windows reach before the first data hour")
    print("Modelling table:", model_df.shape, "| features:", len(NUM_FEATURES) + len(CAT_FEATURES))
    return model_df


def chronological_split(model_df: pd.DataFrame, train_end: pd.Timestamp = TRAIN_END):
    """
    Split by time instead of randomly: everything before train_end is training data,
    everything from train_end on is test data. A random split would let the model see
    the hours right before and after each test hour.
    Rows are sorted by time so TimeSeriesSplit can be used for cross-validation on the training set.

    :param model_df: modelling table
    :param train_end: first timestamp of the test period
    :return: (train, test)
    """
    model_df = model_df.sort_values(["ts", "region"]).reset_index(drop=True)
    train = model_df[model_df.ts < train_end]
    test = model_df[model_df.ts >= train_end]
    print(f"Train: {train.shape}  {train.ts.min():%Y-%m-%d} → {train.ts.max():%Y-%m-%d}")
    print(f"Test : {test.shape}  {test.ts.min():%Y-%m-%d} → {test.ts.max():%Y-%m-%d}")
    return train, test
