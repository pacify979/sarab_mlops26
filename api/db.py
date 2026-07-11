"""SQLite logging of every inference request/response.

Each prediction is persisted so we can (a) audit historical performance and
(b) reconstruct the "current" data window that EvidentlyAI compares against the
training reference to detect data / prediction drift.

Features are stored as a JSON blob to keep the schema robust to feature changes;
the monitoring layer expands them with pandas.json_normalize.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "monitoring" / "predictions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL,          -- ISO timestamp of the request
    features_json       TEXT NOT NULL,          -- input feature vector as JSON
    failure_probability REAL NOT NULL,
    failure_predicted   INTEGER NOT NULL        -- 0/1
);
"""


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_SCHEMA)


def log_prediction(ts: str, features: dict, proba: float, predicted: bool) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO predictions (ts, features_json, failure_probability, failure_predicted) "
            "VALUES (?, ?, ?, ?)",
            (ts, json.dumps(features), float(proba), int(predicted)),
        )


def count_rows() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
