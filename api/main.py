"""FastAPI service for real-time failure prediction on AMR-fleet telemetry.

Endpoints
  GET  /health   -> liveness + model status
  POST /predict  -> failure probability for one telemetry feature vector
  GET  /stats    -> number of logged predictions (quick monitoring sanity check)

Every prediction is logged to SQLite (api/db.py) for historical analysis and to
feed the EvidentlyAI drift monitor.

Run:  uvicorn api.main:app --reload      (from project root, venv activated)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api import db
from api.schemas import PredictionResponse, TelemetryFeatures
from src import config as C

# Decision threshold for turning probability into a 0/1 alert.
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

_state: dict = {}   # holds the loaded model + feature order


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the pruned production model + its expected feature order at startup.
    _state["model"] = joblib.load(C.MODELS_DIR / "model.joblib")
    _state["feature_columns"] = joblib.load(C.MODELS_DIR / "feature_columns.joblib")
    db.init_db()
    yield
    _state.clear()


app = FastAPI(title="AMR Fleet Failure-Prediction API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state, "threshold": THRESHOLD}


@app.get("/stats")
def stats() -> dict:
    return {"logged_predictions": db.count_rows()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TelemetryFeatures) -> PredictionResponse:
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feat_dict = features.model_dump()
    # Build a single-row frame with exactly the columns the model was trained on.
    row = pd.DataFrame([feat_dict])[_state["feature_columns"]]

    proba = float(model.predict_proba(row)[0, 1])
    predicted = proba >= THRESHOLD

    db.log_prediction(
        ts=datetime.now(timezone.utc).isoformat(),
        features=feat_dict,
        proba=proba,
        predicted=predicted,
    )
    return PredictionResponse(
        failure_probability=proba,
        failure_predicted=predicted,
        threshold=THRESHOLD,
    )
