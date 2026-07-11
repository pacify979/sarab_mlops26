"""Train a failure-prediction model and demonstrate pruning-based compression.

Two models are trained on identical data:
  * baseline  : RandomForest, no pruning (ccp_alpha = 0)
  * pruned    : RandomForest with cost-complexity pruning (ccp_alpha > 0)

Cost-complexity pruning (Breiman) removes the weakest branches of each tree,
shrinking the model. We report the trade-off that matters for a real-time API:
    model size on disk  +  inference latency   vs.   predictive quality (PR-AUC).
The pruned model is saved as the production artifact for the FastAPI service.

IMPORTANT: the train/test split is by TIME, not random. Telemetry is a time
series; a random split would leak future readings into training and inflate
scores. We train on the earlier period and test on the later one.

Run:  python -m src.train      (from the project root, venv activated)
"""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config as C

# Columns that are identifiers/target, not model inputs.
DROP_COLS = ["datetime", "machineID", C.LABEL_COL]
CATEGORICAL = ["model"]

TEST_FRACTION = 0.2           # last 20% of the timeline is the test set
CCP_ALPHA_GRID = [1e-5, 5e-5, 1e-4, 5e-4]   # candidate pruning strengths
N_ESTIMATORS = 100
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: earlier rows train, later rows test."""
    df = df.sort_values("datetime").reset_index(drop=True)
    cutoff = int(len(df) * (1 - TEST_FRACTION))
    split_time = df.loc[cutoff, "datetime"]
    train = df[df["datetime"] < split_time]
    test = df[df["datetime"] >= split_time]
    return train, test


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=DROP_COLS)
    y = df[C.LABEL_COL].to_numpy()
    return X, y


def build_pipeline(feature_cols: list[str], ccp_alpha: float) -> Pipeline:
    """Preprocessing (one-hot the model column) + RandomForest with pruning."""
    numeric = [c for c in feature_cols if c not in CATEGORICAL]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", numeric),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",   # the positive class is only ~2%
        ccp_alpha=ccp_alpha,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "pr_auc": average_precision_score(y_test, proba),   # key metric under imbalance
        "roc_auc": roc_auc_score(y_test, proba),
    }


def model_size_mb(pipe: Pipeline, path) -> float:
    joblib.dump(pipe, path)
    return path.stat().st_size / 1e6


def latency_ms(pipe: Pipeline, X_test: pd.DataFrame, n: int = 2000) -> float:
    """Median single-row inference latency — what a real-time API experiences."""
    sample = X_test.iloc[:n]
    times = []
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        t0 = time.perf_counter()
        pipe.predict(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def total_nodes(pipe: Pipeline) -> int:
    forest = pipe.named_steps["clf"]
    return int(sum(est.tree_.node_count for est in forest.estimators_))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.FEATURES_PARQUET)
    train_df, test_df = time_split(df)
    X_train, y_train = make_xy(train_df)
    X_test, y_test = make_xy(test_df)
    feature_cols = list(X_train.columns)

    print(f"Train: {len(X_train):,} rows ({y_train.mean()*100:.2f}% pos) "
          f"| Test: {len(X_test):,} rows ({y_test.mean()*100:.2f}% pos)")
    print(f"Split at: {test_df['datetime'].min()}\n")

    # --- Baseline (no pruning) ---
    print("Training baseline (ccp_alpha=0)...")
    baseline = build_pipeline(feature_cols, ccp_alpha=0.0).fit(X_train, y_train)
    base_metrics = evaluate(baseline, X_test, y_test)

    # --- Pick pruning strength: strongest pruning that keeps PR-AUC within 2% ---
    print("Searching pruning strength (ccp_alpha)...")
    best = {"alpha": 0.0, "pipe": baseline, "metrics": base_metrics}
    for alpha in CCP_ALPHA_GRID:
        pipe = build_pipeline(feature_cols, ccp_alpha=alpha).fit(X_train, y_train)
        m = evaluate(pipe, X_test, y_test)
        keeps_quality = m["pr_auc"] >= base_metrics["pr_auc"] * 0.98
        print(f"  alpha={alpha:<8g} PR-AUC={m['pr_auc']:.4f} "
              f"{'(ok)' if keeps_quality else '(too lossy)'}")
        if keeps_quality:
            best = {"alpha": alpha, "pipe": pipe, "metrics": m}   # grid ascends -> keep strongest
    pruned = best["pipe"]

    # --- Compression report: baseline vs pruned ---
    base_path = C.MODELS_DIR / "model_baseline.joblib"
    prod_path = C.MODELS_DIR / "model.joblib"   # pruned = production artifact
    base_size = model_size_mb(baseline, base_path)
    pruned_size = model_size_mb(pruned, prod_path)

    print("\n=== Baseline vs Pruned ===")
    print(f"{'metric':<18}{'baseline':>14}{'pruned':>14}")
    print(f"{'ccp_alpha':<18}{0.0:>14g}{best['alpha']:>14g}")
    print(f"{'PR-AUC':<18}{base_metrics['pr_auc']:>14.4f}{best['metrics']['pr_auc']:>14.4f}")
    print(f"{'ROC-AUC':<18}{base_metrics['roc_auc']:>14.4f}{best['metrics']['roc_auc']:>14.4f}")
    print(f"{'tree nodes':<18}{total_nodes(baseline):>14,}{total_nodes(pruned):>14,}")
    print(f"{'size (MB)':<18}{base_size:>14.2f}{pruned_size:>14.2f}")
    print(f"{'latency (ms)':<18}{latency_ms(baseline, X_test):>14.3f}{latency_ms(pruned, X_test):>14.3f}")

    # Persist the exact feature order for the serving layer to validate against.
    joblib.dump(feature_cols, C.MODELS_DIR / "feature_columns.joblib")
    print(f"\nProduction model saved -> {prod_path}")
    print(f"Feature schema saved   -> {C.MODELS_DIR / 'feature_columns.joblib'}")


if __name__ == "__main__":
    main()
