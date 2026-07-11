"""Central paths and constants for the AMR-fleet monitoring project.

The dataset is the Microsoft Azure Predictive Maintenance set, used here as a
*structural proxy* for an AMR (autonomous mobile robot) fleet: each machine is a
mobile unit emitting multivariate operational telemetry over time, with
component-failure outcome labels. Feature semantics map cleanly onto robot
telemetry (e.g. voltage->motor current, vibration->chassis vibration).
"""
from pathlib import Path

# --- Directories ---
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# --- Raw files (as downloaded from Kaggle) ---
TELEMETRY_CSV = DATASET_DIR / "PdM_telemetry.csv"
ERRORS_CSV = DATASET_DIR / "PdM_errors.csv"
MAINT_CSV = DATASET_DIR / "PdM_maint.csv"
FAILURES_CSV = DATASET_DIR / "PdM_failures.csv"
MACHINES_CSV = DATASET_DIR / "PdM_machines.csv"

# --- Processed output ---
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"

# --- Modeling constants ---
SENSORS = ["volt", "rotate", "pressure", "vibration"]
ERROR_TYPES = ["error1", "error2", "error3", "error4", "error5"]

FEATURE_CADENCE_H = 3      # emit one feature row every 3 hours
SHORT_WINDOW_H = 3         # short rolling window
LONG_WINDOW_H = 24         # long rolling window
LOOKAHEAD_H = 24           # predict a failure within the next 24 hours

# Label column produced by data_prep.py: 1 if any component fails within
# LOOKAHEAD_H hours of the feature timestamp, else 0. This is the AMR
# "will this unit fail to complete its task" analog.
LABEL_COL = "failure_within_24h"
