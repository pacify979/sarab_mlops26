# AMR Fleet Monitoring

An MLOps architecture for monitoring an AMR (Autonomous Mobile Robot) fleet
in a logistics center. It predicts, from streaming fleet telemetry, whether a
mobile unit is about to fail its task (component failure within 24h), serves the
model over a REST API, and continuously monitors the live data and predictions
for drift.

The dataset is the [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
set, used as a **structural proxy** for AMR telemetry: a fleet of 100 mobile
units emitting multivariate operational telemetry over time, with failure
outcome labels. (See the project guide for the full analogy and feature mapping.)

## MLOps components implemented

- **Serving & deployment**: FastAPI real-time inference, model compression via
  cost-complexity pruning (38× smaller, no accuracy loss), and a Docker image.
- **Monitoring & observability**: EvidentlyAI data-drift, prediction-drift,
  and data-quality reports, with every request logged to SQLite for
  historical analysis and automated retraining triggers.

## Pipeline

```
raw CSVs → src/data_prep.py → features.parquet → src/train.py → models/model.joblib
        → api/main.py (serving + SQLite log) → monitoring/monitor.py → drift reports
```

## Results

PR-AUC 0.82 · pruning makes the model **38× smaller** with no accuracy loss ·
the drift monitor correctly **flags an injected sensor fault** while staying
**quiet on healthy traffic**.

## Quickstart

```bash
# one-time system prerequisite (Debian/Ubuntu)
sudo apt install python3.10-venv

# environment
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# place the 5 Kaggle PdM_*.csv files in dataset/, then:
python -m src.data_prep      # build features -> data/processed/features.parquet
python -m src.train          # train + prune  -> models/model.joblib
uvicorn api.main:app --port 8000   # serve -> http://127.0.0.1:8000/docs

# monitoring (new terminal, venv active, API running): 
# run the monitor right after matching replay
python -m monitoring.replay              && python -m monitoring.monitor --label healthy
python -m monitoring.replay --drift      && python -m monitoring.monitor --label drift
```

Containerized serving (alternative to running `uvicorn` directly):

```bash
docker build -t amr-fleet-api .
# mount monitoring/ so the request log lands on the host (needed for the monitor)
docker run --rm -p 8000:8000 -v "$(pwd)/monitoring:/app/monitoring" amr-fleet-api
# -> http://127.0.0.1:8000/docs
```

The `-v` mount is required if you want to run the drift monitor against the
container, without it the API logs inside the container and the host-side
monitor sees a stale database.

## Documentation

**[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)**: detailed guide containing every
component, every dependency, design reasoning, and step-by-step run instructions
for every part of the code.

## Project layout

| Path | Purpose |
|------|---------|
| `src/` | Data prep (`data_prep.py`), training + pruning (`train.py`), config |
| `api/` | FastAPI serving (`main.py`), Pydantic schemas, SQLite logging |
| `monitoring/` | Traffic replay (`replay.py`), EvidentlyAI drift monitor (`monitor.py`) |
| `models/` | Trained artifacts (gitignored) |
| `dataset/` | Raw Kaggle CSVs (gitignored) |
| `Dockerfile`, `requirements-api.txt` | Lean serving container |
