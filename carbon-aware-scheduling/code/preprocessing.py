"""
Step 3-4: clean each source, put everything on one hourly UTC grid, merge, and impute.

Issues handled (found in EDA):
 1  Timestamps are strings; CI is naive UTC, prices tz-aware, weather LOCAL time  -> parse everything to tz-aware UTC
 2  Weather uses one fixed UTC offset per file (every step is exactly 1 h)         -> subtract utc_offset_seconds
 2b India is UTC+5:30, so weather/price stamps fall on :30                         -> interpolate onto the full hour
 3  Inconsistent region ids between sources                                        -> folder name is the key
 4  Constant columns (Country, Zone Name, Zone Id, Data Source)                    -> drop
 5  `Data Estimation Method` 86% missing (missing = measured)                      -> keep boolean is_estimated only
 6  CI (direct) is measured at the same time as the target = leakage               -> never used as a feature
 7  Duplicates / missing hours                                                     -> dedupe, re-grid, interpolate gaps <= 6 h
 8  Price spikes and negatives                                                     -> keep negatives, winsorise 0.5/99.5 pct
 9  GB has no price data                                                           -> per-hour train median + price_missing flag
 10 Physical range checks                                                          -> 0 <= CI <= 1500, humidity/cloud in [0,100]
Every statistic learned from the data (percentiles, medians) uses ONLY pre-2024 (training) rows.
"""
import numpy as np
import pandas as pd

from settings import TARGET_RAW, TRAIN_END


def align_to_grid(frame: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Put a time-indexed table onto the hourly UTC grid. If its timestamps are not on the
    full hour (India's :30), the values are linearly interpolated to the full hour instead
    of being dropped in the join.

    :param frame: table indexed by UTC timestamp
    :param grid: hourly UTC timestamps to align to
    :return: table indexed by grid
    """
    if frame.index.isin(grid).all():
        return frame.reindex(grid)
    both = frame.reindex(frame.index.union(grid))
    return both.interpolate(method="time", limit=2, limit_area="inside").reindex(grid)


def clean_carbon_intensity(ci_raw: pd.DataFrame):
    """
    Clean the carbon intensity data: rename columns, parse timestamps, remove duplicates and
    impossible values, add missing hours and interpolate short gaps.

    :param ci_raw: raw carbon intensity rows
    :return: (clean table, full hourly UTC grid used to align the other sources)
    """
    ci = ci_raw.rename(columns={
        "Datetime (UTC)": "ts", TARGET_RAW: "ci", "Carbon Intensity gCO₂eq/kWh (direct)": "ci_direct",
        "Low Carbon Percentage": "low_carbon_pct", "Renewable Percentage": "renewable_pct",
        "Data Estimated": "is_estimated"})
    ci["ts"] = pd.to_datetime(ci["ts"], utc=True)
    ci["is_estimated"] = ci["is_estimated"].astype(str).str.lower().eq("true").astype(int)
    ci = ci[["region", "ts", "ci", "ci_direct", "low_carbon_pct", "renewable_pct", "is_estimated"]]

    n0 = len(ci)
    ci = ci.drop_duplicates(["region", "ts"]).sort_values(["region", "ts"])
    print(f"Duplicates removed: {n0 - len(ci)}")

    bad = ~ci["ci"].between(0, 1500)          # impossible values -> NaN so they get interpolated
    print("CI values outside [0,1500]:", int(bad.sum()))
    ci.loc[bad, "ci"] = np.nan

    full_idx = pd.date_range(ci.ts.min(), ci.ts.max(), freq="h")
    ci = (ci.set_index("ts").groupby("region", group_keys=False)
            .apply(lambda g: g.reindex(full_idx).assign(region=g.name))
            .rename_axis("ts").reset_index())
    print("Missing hours created by re-gridding:", int(ci["ci"].isna().sum()))

    num_cols = ["ci", "ci_direct", "low_carbon_pct", "renewable_pct"]
    ci[num_cols] = ci.groupby("region")[num_cols].transform(lambda s: s.interpolate(limit=6, limit_direction="both"))
    ci["is_estimated"] = ci["is_estimated"].fillna(1).astype(int)
    print("CI clean shape:", ci.shape, "| remaining NaNs:", int(ci[num_cols].isna().sum().sum()))
    return ci, full_idx
