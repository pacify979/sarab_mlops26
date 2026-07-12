"""EvidentlyAI drift + data-quality monitoring for the fleet failure model.

Compares two distributions:
  * reference : the model's training-period window (what 'normal' looked like)
  * current   : recent production requests read back from the SQLite log

Produces:
  * reports/drift_report.html   -- full interactive Evidently report
  * reports/drift_status.json   -- machine-readable verdict (the observability
                                   hook an automated retraining trigger would read)
  * a printed summary + DRIFT / OK verdict

Metrics: DataDriftPreset (per-feature + prediction drift) and DataQualityPreset
(missing values, ranges, etc.). Prediction drift is tracked by mapping the logged
`failure_probability` as the model's prediction column.

Run:  python -m monitoring.monitor      (from project root, venv activated)
"""
from __future__ import annotations

import argparse
import json
import sqlite3

import joblib
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

from api.db import DB_PATH
from src import config as C
from src.train import CATEGORICAL, time_split

REPORTS_DIR = C.ROOT / "reports"
PRED_COL = "failure_probability"
REFERENCE_SAMPLE = 15000   # cap reference size so the report renders quickly
CURRENT_WINDOW = 2000      # analyze only the most recent N logged requests
SHARE_DRIFT_THRESHOLD = 0.30   # alert if >=30% of input features drift


def load_reference() -> tuple[pd.DataFrame, list[str]]:
    """Training-period features + the model's own predicted probabilities."""
    df = pd.read_parquet(C.FEATURES_PARQUET)
    train_df, _ = time_split(df)
    feat_cols = joblib.load(C.MODELS_DIR / "feature_columns.joblib")
    model = joblib.load(C.MODELS_DIR / "model.joblib")

    ref = train_df[feat_cols].copy()
    if len(ref) > REFERENCE_SAMPLE:
        ref = ref.sample(REFERENCE_SAMPLE, random_state=42)
    ref[PRED_COL] = model.predict_proba(ref[feat_cols])[:, 1]
    return ref, feat_cols


def load_current(feat_cols: list[str]) -> pd.DataFrame:
    """The most recent CURRENT_WINDOW production requests from the SQLite log.

    Only a bounded, recent window is analyzed (not the whole table), so older
    traffic can't contaminate the comparison and no manual DB reset is needed.
    This mirrors real monitoring: 'look at the last N requests'.
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = pd.read_sql(
            "SELECT features_json, failure_probability FROM predictions "
            "ORDER BY id DESC LIMIT ?",
            conn,
            params=(CURRENT_WINDOW,),
        )
    if rows.empty:
        raise SystemExit(
            "No logged predictions found. Start the API and run "
            "`python -m monitoring.replay` first."
        )
    feats = pd.json_normalize(rows["features_json"].map(json.loads))
    feats = feats[feat_cols]
    feats[PRED_COL] = rows["failure_probability"].to_numpy()
    return feats


def summarize(report_dict: dict) -> dict:
    """Pull the headline drift numbers out of Evidently's result dict."""
    out = {"dataset_drift": None, "drifted_columns": None,
           "share_drifted": None, "prediction_drift": None}
    for metric in report_dict["metrics"]:
        name, result = metric["metric"], metric["result"]
        if name == "DatasetDriftMetric":
            out["dataset_drift"] = result.get("dataset_drift")
            out["drifted_columns"] = result.get("number_of_drifted_columns")
            out["share_drifted"] = result.get("share_of_drifted_columns")
        elif name == "DataDriftTable":
            cols = result.get("drift_by_columns", {})
            if PRED_COL in cols:
                out["prediction_drift"] = {
                    "drift_detected": cols[PRED_COL].get("drift_detected"),
                    "drift_score": cols[PRED_COL].get("drift_score"),
                }
    return out


def decide(summary: dict) -> tuple[bool, list[str]]:
    """Observability rule: what would trigger a retraining alert, and why."""
    reasons = []
    if summary["dataset_drift"]:
        reasons.append("dataset-level drift")
    if (summary["share_drifted"] or 0) >= SHARE_DRIFT_THRESHOLD:
        reasons.append(f">={SHARE_DRIFT_THRESHOLD:.0%} of features drifted")
    pd_ = summary["prediction_drift"]
    if pd_ and pd_["drift_detected"]:
        reasons.append("prediction (output) drift")
    return (len(reasons) > 0), reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Evidently drift monitor.")
    parser.add_argument("--label", default="",
                        help="suffix for report filenames, e.g. 'healthy' or 'drift'")
    args = parser.parse_args()
    suffix = f"_{args.label}" if args.label else ""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ref, feat_cols = load_reference()
    cur = load_current(feat_cols)
    numeric = [c for c in feat_cols if c not in CATEGORICAL]

    mapping = ColumnMapping(
        prediction=PRED_COL,
        numerical_features=numeric,
        categorical_features=CATEGORICAL,
    )

    print(f"Reference rows: {len(ref):,} | Current rows: {len(cur):,}")
    print("Running Evidently (data drift + data quality)...")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=mapping)

    html_path = REPORTS_DIR / f"drift_report{suffix}.html"
    report.save_html(str(html_path))
    summary = summarize(report.as_dict())
    drift_flag, reasons = decide(summary)
    summary["drift_alert"] = drift_flag
    summary["reasons"] = reasons

    status_path = REPORTS_DIR / f"drift_status{suffix}.json"
    status_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Drift summary ===")
    print(f"Dataset drift detected : {summary['dataset_drift']}")
    print(f"Drifted columns        : {summary['drifted_columns']} "
          f"({(summary['share_drifted'] or 0) * 100:.1f}% of columns)")
    if summary["prediction_drift"] is not None:
        pd_ = summary["prediction_drift"]
        print(f"Prediction drift       : {pd_['drift_detected']} "
              f"(score={pd_['drift_score']:.4f})")
    verdict = (f"DRIFT DETECTED -> consider retraining ({', '.join(reasons)})"
               if drift_flag else "OK -> no significant drift")
    print(f"\nObservability verdict  : {verdict}")
    print(f"HTML report            : {html_path}")
    print(f"Machine-readable status: {status_path}")


if __name__ == "__main__":
    main()
