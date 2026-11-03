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

