"""
Step 2: exploratory data analysis on the RAW data (before cleaning). Prints summaries and saves figures.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from settings import FIG_DIR, TARGET_RAW


def summarize_raw(ci_raw: pd.DataFrame, w_raw: pd.DataFrame, p_raw: pd.DataFrame) -> None:
    """
    Print data types, missing values and basic sanity checks for the three raw sources.
    These checks are what the cleaning steps in preprocessing.py are based on.

    :param ci_raw: raw carbon intensity rows
    :param w_raw: raw weather rows
    :param p_raw: raw electricity price rows
    :return: None
    """
    print("dtypes:\n", ci_raw.dtypes, "\n")
    print("Missing values per column:\n", ci_raw.isna().sum(), "\n")
    print("Rows marked as estimated:", round(ci_raw["Data Estimated"].mean() * 100, 1), "%")
    print(ci_raw["Data Estimation Method"].value_counts(dropna=False), "\n")
    print("Constant / near-constant text columns:",
          {c: ci_raw[c].nunique() for c in ["Country", "Zone Name", "Zone Id", "Data Source"]})

    print("\nPrice summary (USD/MWh):\n", p_raw["Price (USD/MWh)"].describe().round(1).to_frame().T)
    print("Negative prices:", int((p_raw["Price (USD/MWh)"] < 0).sum()),
          "| prices > 1000:", int((p_raw["Price (USD/MWh)"] > 1000).sum()))
    print("Weather nulls:", w_raw.isna().sum().to_dict())
    print("Weather physical-range violations:",
          {"humidity": int((~w_raw.relative_humidity_2m.between(0, 100)).sum()),
           "cloudcover": int((~w_raw.cloudcover.between(0, 100)).sum()),
           "wind<0": int((w_raw.windspeed_10m < 0).sum()),
           "temp outside [-60,60]": int((~w_raw.temperature_2m.between(-60, 60)).sum())})
