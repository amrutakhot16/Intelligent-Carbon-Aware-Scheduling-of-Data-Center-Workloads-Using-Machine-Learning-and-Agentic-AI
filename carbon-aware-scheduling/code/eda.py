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


def plot_region_boxplot(ci_raw: pd.DataFrame, fig_dir=FIG_DIR) -> pd.DataFrame:
    """
    Boxplot of carbon intensity per region, sorted by the mean.
    Shows how much cleaner some grids are than others, i.e. how much can be saved
    by choosing where a job runs.

    :param ci_raw: raw carbon intensity rows
    :param fig_dir: folder to save the figure in
    :return: per-region summary table (count, mean, std, min, max)
    """
    stats = (ci_raw.groupby("region")[TARGET_RAW].agg(["count", "mean", "std", "min", "max"])
             .round(1).sort_values("mean"))
    print(stats)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.boxplot(data=ci_raw, x="region", y=TARGET_RAW, order=stats.index.tolist(), ax=ax, fliersize=1, color="#7fb3a8")
    ax.set(title="Carbon intensity by region, 2021–2024 (sorted by mean)", xlabel="", ylabel="gCO₂eq/kWh")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(fig_dir / "eda_boxplot_regions.png", dpi=120); plt.close()
    return stats


def plot_daily_profiles(ci_raw: pd.DataFrame, regions: list, region_tz: dict, fig_dir=FIG_DIR) -> None:
    """
    Plot the average carbon intensity for each hour of the day (local time) for a few
    regions, plus two weeks of hourly data for Germany. Shows how much can be saved
    by choosing when a job runs.

    :param ci_raw: raw carbon intensity rows
    :param regions: list of region ids
    :param region_tz: {region: timezone name}, used to convert UTC to local time
    :param fig_dir: folder to save the figure in
    :return: None
    """
    tmp = ci_raw[["Datetime (UTC)", "region", TARGET_RAW]].copy()
    tmp["ts"] = pd.to_datetime(tmp["Datetime (UTC)"], utc=True)
    tmp["local_hour"] = np.nan
    for r in regions:
        m = tmp.region == r
        tmp.loc[m, "local_hour"] = tmp.loc[m, "ts"].dt.tz_convert(region_tz[r]).dt.hour

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for r in ["DE-LU", "US-CAL-CISO", "GB", "US-NY-NYIS", "CL-SIC", "IN-WE"]:
        prof = tmp[tmp.region == r].groupby("local_hour")[TARGET_RAW].mean()
        axes[0].plot(prof.index, prof.values, marker="o", ms=3, label=r)
    axes[0].set(title="Average daily profile (local time)", xlabel="Local hour", ylabel="gCO₂eq/kWh")
    axes[0].legend(fontsize=8)
    de = tmp[(tmp.region == "DE-LU") & (tmp.ts >= "2023-06-01") & (tmp.ts < "2023-06-15")]
    axes[1].plot(de.ts, de[TARGET_RAW], color="#2b6f8f")
    axes[1].set(title="Germany, first two weeks of June 2023", ylabel="gCO₂eq/kWh")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(fig_dir / "eda_daily_profiles.png", dpi=120); plt.close()
