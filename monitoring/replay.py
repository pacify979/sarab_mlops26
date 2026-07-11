"""Replay real later-period telemetry through the live API as 'production' traffic.

This populates the SQLite log (via the API's normal logging path) with a window
of genuine post-training data, which the monitor then compares against the
training-period reference to detect natural drift.

Requires the API to be running:  uvicorn api.main:app --port 8000
Run:  python -m monitoring.replay      (from project root, venv activated)
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

from src import config as C
from src.train import time_split

API_URL = "http://127.0.0.1:8000/predict"
N_REQUESTS = 2000       # sample size of the 'current' production window
RANDOM_STATE = 42

# Synthetic fleet-wide fault used as a POSITIVE CONTROL for the drift monitor:
# elevated vibration (and its variability) plus higher pressure, as a robot
# developing a mechanical fault would exhibit. Multiplicative shifts per column.
DRIFT_SHIFTS = {
    "vibration_mean_3h": 1.20, "vibration_mean_24h": 1.20,
    "vibration_std_3h": 1.40, "vibration_std_24h": 1.40,
    "pressure_mean_3h": 1.10, "pressure_mean_24h": 1.10,
}


def apply_drift(sample: pd.DataFrame) -> pd.DataFrame:
    sample = sample.copy()
    for col, factor in DRIFT_SHIFTS.items():
        sample[col] = sample[col] * factor
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay production traffic to the API.")
    parser.add_argument("--drift", action="store_true",
                        help="inject a synthetic sensor fault (positive control)")
    args = parser.parse_args()

    df = pd.read_parquet(C.FEATURES_PARQUET)
    _, later = time_split(df)   # the later (post-training) period
    sample = later.drop(columns=["datetime", "machineID", C.LABEL_COL])
    sample = sample.sample(min(N_REQUESTS, len(sample)), random_state=RANDOM_STATE)
    if args.drift:
        sample = apply_drift(sample)
        print("INJECTED DRIFT: elevated vibration + pressure (positive control)")

    sent = 0
    for _, row in sample.iterrows():
        payload = json.dumps(row.to_dict(), default=float).encode()
        req = urllib.request.Request(
            API_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            sent += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Request failed after {sent} sent: {exc}")
            print("Is the API running?  uvicorn api.main:app --port 8000")
            return
        if sent % 250 == 0:
            print(f"  ...{sent} sent")

    print(f"Replayed {sent} production requests -> logged to SQLite.")


if __name__ == "__main__":
    main()
