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


def clean_weather(w_raw: pd.DataFrame, w_offset: dict, full_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Convert the weather timestamps from local time to UTC, align them to the carbon intensity
    grid, fill short gaps and clip humidity and cloud cover to 0-100.

    :param w_raw: raw weather rows
    :param w_offset: {region: set of UTC offsets in seconds}
    :param full_idx: hourly UTC grid from clean_carbon_intensity()
    :return: clean weather table with region, ts and the four weather columns
    """
    steps = w_raw.assign(t=pd.to_datetime(w_raw["time"])).groupby("region")["t"].diff().dropna()
    print("Time steps between consecutive weather rows:", steps.value_counts().to_dict())
    print("Fixed UTC offsets per region (seconds):", {k: sorted(v) for k, v in w_offset.items()})

    parts = []
    for r, g in w_raw.groupby("region"):
        off = pd.Timedelta(seconds=list(w_offset[r])[0])
        ts = (pd.to_datetime(g["time"]) - off).dt.tz_localize("UTC")
        g = g.assign(ts=ts).drop_duplicates("ts").set_index("ts").drop(columns=["time", "region"])
        g = align_to_grid(g, full_idx)
        n_gap = int(g.isna().any(axis=1).sum())
        g = g.interpolate(method="time", limit=6).ffill(limit=12).bfill(limit=12)
        if n_gap:
            print(f"  {r}: filled {n_gap} missing weather hours")
        parts.append(g.assign(region=r))

    weather = pd.concat(parts).rename_axis("ts").reset_index()
    weather = weather.rename(columns={"temperature_2m": "temp_c", "relative_humidity_2m": "humidity",
                                      "cloudcover": "cloud_pct", "windspeed_10m": "wind_kmh"})
    weather["humidity"] = weather["humidity"].clip(0, 100)
    weather["cloud_pct"] = weather["cloud_pct"].clip(0, 100)
    print("Weather clean shape:", weather.shape, "| NaNs:", int(weather.isna().sum().sum()))
    return weather


def clean_prices(p_raw: pd.DataFrame, regions: list, full_idx: pd.DatetimeIndex,
                 train_end: pd.Timestamp = TRAIN_END) -> pd.DataFrame:
    """
    Align the prices to the hourly grid and clip extreme values per region.
    The clipping limits come from the training years only.

    :param p_raw: raw price rows
    :param regions: regions to keep (the ones that have carbon intensity data)
    :param full_idx: hourly UTC grid from clean_carbon_intensity()
    :param train_end: start of the test period; only earlier rows are used for the limits
    :return: clean price table with region, ts and price
    """
    price = p_raw.rename(columns={"Datetime (UTC)": "ts", "Price (USD/MWh)": "price"})[["region", "ts", "price"]]
    price["ts"] = pd.to_datetime(price["ts"], utc=True)
    price = price[price.region.isin(regions)].drop_duplicates(["region", "ts"])
    off_hour = price.ts.dt.minute != 0
    print("Price rows not on the full hour:", price[off_hour].region.value_counts().to_dict())

    price = pd.concat([align_to_grid(g.set_index("ts")[["price"]], full_idx).assign(region=r)
                       for r, g in price.groupby("region")]).rename_axis("ts").reset_index().dropna(subset=["price"])

    q = price[price.ts < train_end].groupby("region")["price"].quantile([0.005, 0.995]).unstack()
    price = price.join(q, on="region")
    n_clip = int(((price.price < price[0.005]) | (price.price > price[0.995])).sum())
    price["price"] = price["price"].clip(price[0.005], price[0.995])
    price = price.drop(columns=[0.005, 0.995])
    print(f"Winsorised {n_clip} extreme price values ({n_clip / len(price):.2%})")
    return price


def merge_sources(ci: pd.DataFrame, weather: pd.DataFrame, price: pd.DataFrame,
                  train_end: pd.Timestamp = TRAIN_END) -> pd.DataFrame:
    """
    Join weather and prices onto the carbon intensity table. Missing prices (GB has none)
    are filled with the median price for that UTC hour in the training years, and a
    price_missing flag records which rows were filled.

    :param ci: clean carbon intensity table
    :param weather: clean weather table
    :param price: clean price table
    :param train_end: start of the test period; only earlier rows are used for the median
    :return: merged hourly table
    """
    df = ci.merge(weather, on=["region", "ts"], how="left").merge(price, on=["region", "ts"], how="left")
    print("Merged shape:", df.shape)
    na = df.isna().sum()
    print("Missing values after merge:\n", na[na > 0])

    df["price_missing"] = df["price"].isna().astype(int)
    df["hour_utc"] = df.ts.dt.hour
    hourly_median = df[df.ts < train_end].groupby("hour_utc")["price"].median()
    df["price"] = df["price"].fillna(df["hour_utc"].map(hourly_median))
    print("After imputation, total NaNs:", int(df.isna().sum().sum()))
    return df


def preprocess(raw: dict) -> pd.DataFrame:
    """
    Run all the cleaning steps and merge the three sources.

    :param raw: dict returned by data_loading.load_all()
    :return: merged hourly table, ready for feature engineering
    """
    ci, full_idx = clean_carbon_intensity(raw["ci_raw"])
    weather = clean_weather(raw["w_raw"], raw["w_offset"], full_idx)
    price = clean_prices(raw["p_raw"], raw["regions"], full_idx)
    return merge_sources(ci, weather, price)

