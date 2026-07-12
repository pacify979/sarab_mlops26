"""Build a labeled, feature-engineered training table from raw PdM telemetry.
1. Load the five raw CSVs.
2. Engineer rolling-window telemetry features (3h and 24h mean/std) per machine.
3. Add 24h rolling error counts per error type.
4. Merge static machine metadata (model, age).
5. Sample one feature row every FEATURE_CADENCE_H hours.
6. Label each row: 1 if any component fails within LOOKAHEAD_H hours, else 0.

Run:  python -m src.data_prep      (from the project root, venv activated)
Output: data/processed/features.parquet
"""
from __future__ import annotations

import pandas as pd

from src import config as C


# Loading
def load_raw() -> dict[str, pd.DataFrame]:
    """Load the five raw CSVs, parsing datetimes."""
    telemetry = pd.read_csv(C.TELEMETRY_CSV, parse_dates=["datetime"])
    errors = pd.read_csv(C.ERRORS_CSV, parse_dates=["datetime"])
    maint = pd.read_csv(C.MAINT_CSV, parse_dates=["datetime"])
    failures = pd.read_csv(C.FAILURES_CSV, parse_dates=["datetime"])
    machines = pd.read_csv(C.MACHINES_CSV)
    return {
        "telemetry": telemetry,
        "errors": errors,
        "maint": maint,
        "failures": failures,
        "machines": machines,
    }

# Feature engineering
def telemetry_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean & std of each sensor over short and long windows.

    Telemetry is hourly, so a window of N hours == N rows. Rolling is computed
    per machine (grouped) so windows never bleed across units.
    """
    telemetry = telemetry.sort_values(["machineID", "datetime"])
    g = telemetry.groupby("machineID")

    feats = telemetry[["datetime", "machineID"]].copy()
    for col in C.SENSORS:
        # Short window (3h)
        feats[f"{col}_mean_{C.SHORT_WINDOW_H}h"] = (
            g[col].rolling(C.SHORT_WINDOW_H, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        feats[f"{col}_std_{C.SHORT_WINDOW_H}h"] = (
            g[col].rolling(C.SHORT_WINDOW_H, min_periods=1).std().reset_index(level=0, drop=True)
        )
        # Long window (24h)
        feats[f"{col}_mean_{C.LONG_WINDOW_H}h"] = (
            g[col].rolling(C.LONG_WINDOW_H, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        feats[f"{col}_std_{C.LONG_WINDOW_H}h"] = (
            g[col].rolling(C.LONG_WINDOW_H, min_periods=1).std().reset_index(level=0, drop=True)
        )
    return feats


def error_features(telemetry: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """Count of each error type over the trailing LONG_WINDOW_H hours.

    Errors are sparse events. One-hot encoder, align to the hourly telemetry
    grid per machine, then take a trailing 24h rolling sum.
    """
    # One-hot error events, one row per (machine, datetime).
    err = pd.get_dummies(errors, columns=["errorID"], prefix="", prefix_sep="")
    err = err.groupby(["machineID", "datetime"], as_index=False).sum()

    # Align onto the full hourly telemetry grid so every hour has a count (0 if none).
    grid = telemetry[["machineID", "datetime"]].copy()
    err = grid.merge(err, on=["machineID", "datetime"], how="left") # attach error flag on every machine hour where it exists
    count_cols = [c for c in C.ERROR_TYPES if c in err.columns]
    err[count_cols] = err[count_cols].fillna(0) # hours with no error become 0

    err = err.sort_values(["machineID", "datetime"])
    g = err.groupby("machineID")
    out = err[["datetime", "machineID"]].copy()
    for col in count_cols:
        out[f"{col}_count_{C.LONG_WINDOW_H}h"] = (
            g[col].rolling(C.LONG_WINDOW_H, min_periods=1).sum().reset_index(level=0, drop=True)
        )
    return out

# Labeling
def add_label(features: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """Label = 1 if a component fails within (t, t + LOOKAHEAD_H] for that machine.

    For each failure event we mark the LOOKAHEAD_H feature timestamps that precede
    it (the window during which the model should have raised a warning).
    """
    features = features.copy()
    features[C.LABEL_COL] = 0
    lookahead = pd.Timedelta(hours=C.LOOKAHEAD_H)

    # Index features by machine for fast per-machine masking.
    for machine_id, fail_group in failures.groupby("machineID"):
        mask_machine = features["machineID"] == machine_id
        machine_times = features.loc[mask_machine, "datetime"]
        positive_idx = pd.Index([], dtype="int64")
        for fail_time in fail_group["datetime"]:
            window = machine_times[
                (machine_times > fail_time - lookahead) & (machine_times <= fail_time)
            ]
            positive_idx = positive_idx.union(window.index)
        features.loc[positive_idx, C.LABEL_COL] = 1
    return features

# Orchestration
def build_features() -> pd.DataFrame:
    raw = load_raw()

    print("Engineering telemetry features...")
    tel_feats = telemetry_features(raw["telemetry"])

    print("Engineering error-count features...")
    err_feats = error_features(raw["telemetry"], raw["errors"])

    print("Merging feature blocks + machine metadata...")
    df = tel_feats.merge(err_feats, on=["machineID", "datetime"], how="left")
    df = df.merge(raw["machines"], on="machineID", how="left")

    # Sample every FEATURE_CADENCE_H hours to reduce redundancy / row count.
    df = df[df["datetime"].dt.hour % C.FEATURE_CADENCE_H == 0].copy()

    # Drop early rows whose long-window std is undefined (single-sample std -> NaN).
    df = df.dropna().reset_index(drop=True)

    print("Labeling (failure within next %dh)..." % C.LOOKAHEAD_H)
    df = add_label(df, raw["failures"])

    return df


def main() -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_features()
    df.to_parquet(C.FEATURES_PARQUET, index=False)

    n_pos = int(df[C.LABEL_COL].sum())
    print("\n=== Done ===")
    print(f"Rows:            {len(df):,}")
    print(f"Feature columns: {df.shape[1] - 3}")  # minus datetime, machineID, label
    print(f"Positives:       {n_pos:,} ({100 * n_pos / len(df):.2f}%)")
    print(f"Saved ->         {C.FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
