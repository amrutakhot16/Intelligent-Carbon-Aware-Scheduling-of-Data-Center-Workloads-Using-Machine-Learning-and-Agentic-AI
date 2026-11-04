"""
Step 12: what the forecasts mean for scheduling.

Simulation on the 2024 test data: every day at 00:00 UTC a 4-hour job with a 24-hour deadline arrives.
Each strategy picks a start hour (0–20) using ONLY its forecast; we score the ACTUAL carbon of the chosen window.
* run_now = start immediately (carbon-unaware)
* oracle  = perfect knowledge of the future (upper bound on savings)
* spatio-temporal = may also move the job to any region (where + when), compared with running now in New York.
"""
import numpy as np
import pandas as pd

from settings import JOB_HOURS, REFERENCE_REGION, RESULTS_DIR, TARGET, WINDOW_HOURS


def window_sums(arr: np.ndarray, job_h: int = JOB_HOURS, window: int = WINDOW_HOURS) -> np.ndarray:
    """
    Total carbon intensity of the job for every possible start hour in the day.

    :param arr: array of shape (n_days, 24) with hourly values
    :param job_h: job length in hours
    :param window: hours available to finish the job
    :return: array of shape (n_days, n_starts); column s is the sum of hours s to s + job_h - 1
    """
    return np.stack([arr[:, s:s + job_h].sum(1) for s in range(window - job_h + 1)], axis=1)

def build_cubes(test: pd.DataFrame, preds: dict, strategies: list) -> dict:
    """
    For each region, compute the window totals for every day and start hour, both from the
    actual values and from each strategy's forecast.

    :param test: test set (needs region, ts and the target column)
    :param preds: {strategy name: test predictions}, in the same row order as test
    :param strategies: names of the strategies to include
    :return: {region: {"days": dates, "actual": (days, starts), strategy: (days, starts), ...}}
    """
    frame = test[["region", "ts", TARGET]].copy()
    for s in strategies:
        frame[s] = preds[s]
    frame["day"] = frame.ts.dt.floor("D")
    frame["h"] = frame.ts.dt.hour

    cubes = {}
    for r, g in frame.groupby("region"):
        piv = {c: g.pivot(index="day", columns="h", values=c).dropna() for c in [TARGET] + strategies}
        days = piv[TARGET].index
        cubes[r] = {"days": days, "actual": window_sums(piv[TARGET].loc[days].values),
                    **{s: window_sums(piv[s].loc[days].values) for s in strategies}}
    return cubes


def time_shift_savings(cubes: dict, strategies: list) -> pd.DataFrame:
    """
    CO₂ saved by moving the job to a different start time, staying in the same region,
    compared with starting it immediately.

    :param cubes: output of build_cubes()
    :param strategies: strategy names to compare
    :return: % saving per region (rows) and strategy (columns), also saved as a CSV
    """
    rows = []
    for r, c in cubes.items():
        actual = c["actual"]
        row = {"region": r, "run_now": actual[:, 0].mean(), "oracle": actual.min(1).mean()}
        for s in strategies:
            row[s] = actual[np.arange(len(actual)), c[s].argmin(1)].mean()
        rows.append(row)
    sched = pd.DataFrame(rows).set_index("region")
    saving = ((1 - sched.div(sched["run_now"], axis=0)) * 100).drop(columns="run_now")
    saving = saving.round(1).sort_values(strategies[-1], ascending=False)
    saving.to_csv(RESULTS_DIR / "scheduling_time_shift_savings.csv")
    print("Average CO₂ reduction vs 'run now' in the same region, %:\n", saving.to_string())
    print("Mean over regions:", saving.mean().round(1).to_dict())
    return saving


def spatio_temporal_savings(cubes: dict, strategies: list, regions: list,
                            reference: str = REFERENCE_REGION, job_h: int = JOB_HOURS) -> pd.DataFrame:
    """
    CO₂ saved when the job can be moved to any region and any start time, compared with
    running it immediately in the reference region (New York by default).

    :param cubes: output of build_cubes()
    :param strategies: strategy names to compare
    :param regions: regions the job may be moved to
    :param reference: region used as the "run now" comparison
    :param job_h: job length in hours, used to turn window totals into an average per hour
    :return: average carbon intensity of the chosen windows and the % reduction, also saved as a CSV
    """
    common = min(len(c["actual"]) for c in cubes.values())
    act_all = np.stack([cubes[r]["actual"][:common] for r in regions], 1)      # days × regions × starts
    out = {f"Run now in {reference}": cubes[reference]["actual"][:common, 0].mean()}
    for s in strategies + ["oracle"]:
        est = act_all if s == "oracle" else np.stack([cubes[r][s][:common] for r in regions], 1)
        flat = est.reshape(common, -1).argmin(1)
        out[f"Best region+time by {s}"] = act_all.reshape(common, -1)[np.arange(common), flat].mean()
    sp = pd.Series(out) / job_h
    table = pd.DataFrame({"avg gCO₂/kWh of chosen window": sp.round(1),
                          f"reduction vs {reference} now %": ((1 - sp / sp.iloc[0]) * 100).round(1)})
    table.to_csv(RESULTS_DIR / "scheduling_region_time_savings.csv")
    print(table.to_string())
    return table


def example_recommendation(cubes: dict, regions: list, model: str = "XGBoost",
                           day: str = "2024-03-14", reference: str = REFERENCE_REGION, job_h: int = JOB_HOURS) -> str:
    """
    Show the recommendation the scheduler would make for one job on one day: where and when
    to run it, how much CO₂ that saves, and the next best options.

    :param cubes: output of build_cubes()
    :param regions: regions the job may be moved to
    :param model: whose forecast to use
    :param day: date of the request (job arrives at 00:00 UTC)
    :param reference: region used as the "run now" comparison
    :param job_h: job length in hours
    :return: the recommendation as text (also printed)
    """
    day = pd.Timestamp(day, tz="UTC")
    d_idx = list(cubes[reference]["days"]).index(day)
    est = np.stack([cubes[r][model][d_idx] for r in regions])
    act = np.stack([cubes[r]["actual"][d_idx] for r in regions])
    ri, si = np.unravel_index(est.argmin(), est.shape)
    ref_now = cubes[reference]["actual"][d_idx, 0] / job_h
    best = act[ri, si] / job_h

    lines = [f"Request on {day:%Y-%m-%d} 00:00 UTC: {job_h}-hour job, must finish within {WINDOW_HOURS} h.",
             f"→ Run in {regions[ri]} starting {si:02d}:00 UTC. Forecast {est[ri, si] / job_h:.0f} g/kWh, "
             f"actual {best:.0f} g/kWh vs {ref_now:.0f} g/kWh running now in {reference} → "
             f"{100 * (1 - best / ref_now):.0f}% less CO₂, finishes by {si + job_h:02d}:00 UTC.",
             "Top-3 alternatives by forecast:"]
    for k in np.argsort(est.ravel())[:3]:
        r_, s_ = np.unravel_index(k, est.shape)
        lines.append(f"   {regions[r_]:12s} start {s_:02d}:00 UTC  forecast {est[r_, s_] / job_h:5.0f}  "
                     f"actual {act[r_, s_] / job_h:5.0f} g/kWh")
    text = "\n".join(lines)
    print(text)
    return text
