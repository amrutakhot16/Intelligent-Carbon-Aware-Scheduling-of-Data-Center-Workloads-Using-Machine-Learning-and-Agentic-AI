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

