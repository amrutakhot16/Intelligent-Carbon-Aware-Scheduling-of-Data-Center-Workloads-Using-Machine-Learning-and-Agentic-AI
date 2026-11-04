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


def plot_feature_correlation(model_df: pd.DataFrame, num_features: list, target: str, fig_dir=FIG_DIR) -> list:
    """
    Plot each feature's correlation with the target and the full feature correlation matrix.
    Used as a feature selection check to find features that carry almost the same information.

    :param model_df: modelling table with all engineered features
    :param num_features: names of the numeric feature columns
    :param target: name of the target column
    :param fig_dir: folder to save the figure in
    :return: list of (feature_a, feature_b, r) for pairs with |r| > 0.97
    """
    corr = model_df[num_features + [target]].corr()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.4]})
    corr[target].drop(target).sort_values().plot.barh(ax=axes[0], color="#4c8c7a")
    axes[0].set(title=f"Pearson correlation with target ({target})", xlabel="r")
    sns.heatmap(corr, cmap="vlag", center=0, ax=axes[1], xticklabels=True, yticklabels=True, cbar_kws={"shrink": .6})
    axes[1].set_title("Feature correlation matrix"); axes[1].tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(fig_dir / "feature_correlation.png", dpi=120); plt.close()

    # check each pair of features once (upper triangle of the matrix)
    high = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(num_features)
            for b in num_features[i + 1:] if abs(corr.loc[a, b]) > 0.97]
    print("Highly collinear pairs (|r| > 0.97):", high)
    return high


def run_eda(raw: dict) -> None:
    """
    Run all the EDA on the raw data: printed summaries and the two raw-data figures.
    The feature correlation plot is called separately, after feature engineering.

    :param raw: dict returned by data_loading.load_all()
    :return: None
    """
    summarize_raw(raw["ci_raw"], raw["w_raw"], raw["p_raw"])
    plot_region_boxplot(raw["ci_raw"])
    plot_daily_profiles(raw["ci_raw"], raw["regions"], raw["region_tz"])
