# AMR Fleet Monitoring — Complete Project Guide

An MLOps architecture for monitoring the performance of an **AMR (Autonomous
Mobile Robot) fleet** in a logistics center. This document explains **every**
component, **every** dependency, the reasoning behind each design decision, and
**exactly how to run every part** of the code — from a clean checkout to a
running, monitored, containerized service.

> **Read this first if you are grading, presenting, or re-running the project.**
> Every command is copy-paste ready and assumes you start from the project root:
> `/home/sarab/Projects/sarab_mlops26`.

---

## Table of contents

1. [What the project does](#1-what-the-project-does)
2. [The dataset and the AMR analogy](#2-the-dataset-and-the-amr-analogy)
3. [Repository structure](#3-repository-structure)
4. [The Python environment: what a venv is and why we use one](#4-the-python-environment-what-a-venv-is-and-why-we-use-one)
5. [Dependencies explained, package by package](#5-dependencies-explained-package-by-package)
6. [Stage 1 — Data preparation (`src/data_prep.py`)](#6-stage-1--data-preparation-srcdata_preppy)
7. [Stage 2 — Training and pruning (`src/train.py`)](#7-stage-2--training-and-pruning-srctrainpy)
8. [Stage 3 — Model serving (`api/`)](#8-stage-3--model-serving-api)
9. [Stage 4 — Monitoring and observability (`monitoring/`)](#9-stage-4--monitoring-and-observability-monitoring)
10. [Stage 5 — Docker](#10-stage-5--docker)
11. [End-to-end quickstart (clean machine → running system)](#11-end-to-end-quickstart)
12. [Design decisions and FAQ](#12-design-decisions-and-faq)
13. [Mapping to the theory](#13-mapping-to-the-theory)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What the project does

The system predicts, from streaming fleet telemetry, **whether a mobile unit is
about to fail to complete its task** (a component failure within the next 24
hours), and continuously **monitors** the live data and the model's predictions
for drift.

It implements two MLOps components end-to-end:

- **Model deployment & serving** — a FastAPI REST service for real-time
  inference, a compressed (pruned) model to reduce deployment footprint, and a
  Docker image for portable deployment.
- **Model monitoring** — an EvidentlyAI-based system that detects **data drift**
  and **prediction drift** and checks input **data quality**, logging every
  request to a SQLite database for historical analysis.

The full pipeline is:

```
raw telemetry CSVs
      │  src/data_prep.py      (feature engineering + labeling)
      ▼
data/processed/features.parquet
      │  src/train.py          (time-split training + cost-complexity pruning)
      ▼
models/model.joblib           (pruned production model)
      │  api/main.py           (FastAPI serving; logs every request)
      ▼
monitoring/predictions.db      (SQLite request log)
      │  monitoring/monitor.py  (EvidentlyAI drift + data-quality reports)
      ▼
reports/*.html + *.json        (drift reports + machine-readable verdict)
```

---

## 2. The dataset and the AMR analogy

### 2.1 Why a proxy dataset

Real AMR fleet telemetry is almost always proprietary; public AMR datasets are
tiny and lack **operational outcome labels**. We therefore use a **structural
proxy**: a dataset that shares the same *monitoring topology* — a fleet of
mobile units emitting multivariate operational telemetry over time, with task
outcome labels — even though the units are industrial machines rather than
robots.

The MLOps techniques demonstrated here (serving, drift detection, observability)
depend only on the **statistical behaviour of the signals**, not on whether a
feature is a robot's battery status or a machine's vibration reading. The
analogy is justified at the level of *signal topology*, and the feature mapping
below makes the correspondence explicit.

### 2.2 The dataset: Microsoft Azure Predictive Maintenance

Downloaded from Kaggle
(`arnabbiswas1/microsoft-azure-predictive-maintenance`) and unzipped into
`dataset/`. Five CSV files:

| File | Rows | Contents |
|------|------|----------|
| `PdM_telemetry.csv` | 876,100 | Hourly readings of 4 sensors (`volt`, `rotate`, `pressure`, `vibration`) for 100 machines, all of 2015 |
| `PdM_errors.csv` | 3,919 | Non-fatal error events (`error1`–`error5`) with timestamps |
| `PdM_maint.csv` | 3,286 | Component replacement (maintenance) events (`comp1`–`comp4`) |
| `PdM_failures.csv` | 761 | Component **failures** (`comp1`–`comp4`) — the outcome we predict |
| `PdM_machines.csv` | 100 | Static metadata per machine: `model`, `age` |

**Structure:** 100 units × 8,761 hourly telemetry rows × 4 sensors, over a full
year, with real failure outcome labels. This is exactly the "fleet of mobile
units emitting operational telemetry over time" topology we need.

### 2.3 Feature mapping (AMR ↔ dataset)

| Dataset signal | AMR analog | Role |
|---|---|---|
| `volt`, `rotate`, `pressure`, `vibration` | motor current / wheel odometry / actuator load / chassis vibration | drift-monitored input features |
| machine `age`, `model` | robot age, model variant | static/segment features |
| `error1`–`error5` | robot fault codes | data-quality + categorical signals |
| component failure (`comp1`–`comp4`) | subsystem fault → task abort | **target label** (task success/failure) |
| hourly cadence, 100 units | telemetry heartbeat, fleet of N robots | monitoring topology |

### 2.4 Why not the "Logistics & Supply Chain" dataset

The obvious narrative match (a truck-fleet logistics dataset) was rejected
because it is largely **synthetic**, has weak per-vehicle temporal structure
(making drift detection an artifact rather than a real measurement), and ships
**pre-computed risk columns** (`Disruption Likelihood Score`, `Risk
Classification`) that cause **target leakage** — the model would trivially
reproduce a score generated from the same features. The Azure set forces us to
build the label ourselves, which is honest supervised learning and gives the
drift monitor a real signal to watch.

---

## 3. Repository structure

```
sarab_mlops26/
├── dataset/                     # raw Kaggle CSVs (gitignored, ~108 MB)
│   ├── PdM_telemetry.csv
│   ├── PdM_errors.csv
│   ├── PdM_maint.csv
│   ├── PdM_failures.csv
│   └── PdM_machines.csv
├── src/                         # data + modeling code
│   ├── __init__.py
│   ├── config.py                # all paths + modeling constants
│   ├── data_prep.py             # feature engineering + labeling
│   └── train.py                 # training + cost-complexity pruning
├── api/                         # serving layer
│   ├── __init__.py
│   ├── main.py                  # FastAPI app (health / predict / stats)
│   ├── schemas.py               # Pydantic request/response models
│   └── db.py                    # SQLite request logging
├── monitoring/                  # monitoring layer
│   ├── __init__.py
│   ├── replay.py                # replays traffic to the API (± injected drift)
│   ├── monitor.py               # EvidentlyAI drift + data-quality reports
│   └── predictions.db           # SQLite log (gitignored, created at runtime)
├── models/                      # trained artifacts (gitignored)
│   ├── model.joblib             # pruned PRODUCTION model
│   ├── model_baseline.joblib    # unpruned baseline (for comparison)
│   └── feature_columns.joblib   # exact feature order for the API
├── data/processed/              # feature output (gitignored)
│   └── features.parquet
├── reports/                     # Evidently outputs (gitignored)
│   ├── drift_report_healthy.html
│   ├── drift_report_drift.html
│   ├── drift_status_healthy.json
│   └── drift_status_drift.json
├── docs/
│   └── PROJECT_GUIDE.md         # this file
├── requirements.txt             # full dev environment
├── requirements-api.txt         # lean serving-only deps (for Docker)
├── Dockerfile                   # serving container
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 4. The Python environment: what a venv is and why we use one

### 4.1 What is a virtual environment (venv)?

A **virtual environment** is an isolated Python installation that lives inside
the project (in the `venv/` folder). Instead of installing packages system-wide,
where they can clash with other projects or the OS's own Python, every package
goes into this local folder.

Benefits, and why each matters for this project:

- **Reproducibility** — the exact set of packages is listed in
  `requirements.txt`. Anyone (or the Docker build) can recreate the identical
  environment. MLOps is fundamentally about reproducibility.
- **No conflicts** — this project can use pandas 2.3 while another project uses
  pandas 1.x; they never interfere.
- **Clean removal** — delete `venv/` and every dependency is gone, with the
  system Python untouched.
- **Parity with production** — the same `pip install -r requirements-api.txt`
  runs in the Docker image, so "works on my machine" becomes "works in the
  container".

### 4.2 Why `venv` and not conda / Poetry / uv

`venv` is part of the Python standard library — nothing extra to install — and
maps one-to-one onto the Docker workflow (`pip install -r requirements.txt`).
Conda and Poetry are heavier tools that solve dependency-resolution and
binary-packaging problems this project does not have. For a focused MLOps
service, `venv` + `requirements.txt` is the simplest correct choice.

### 4.3 One-time system prerequisite (Debian/Ubuntu)

On Debian/Ubuntu, creating a venv requires the `python3-venv` system package
(it provides `ensurepip`). If `python3 -m venv venv` fails with *"ensurepip is
not available"*, install it once (requires your password):

```bash
sudo apt install python3.10-venv
```

### 4.4 Creating and using the environment

```bash
# from the project root
python3 -m venv venv           # create the environment (one time)
source venv/bin/activate       # activate it (each new shell)
pip install --upgrade pip
pip install -r requirements.txt # install all dependencies

# when finished:
deactivate
```

Once activated, your shell prompt shows `(venv)` and `python`/`pip` refer to the
isolated environment. **Every run command in this guide assumes the venv is
active.** If you prefer not to activate, prefix commands with the venv's
interpreter instead, e.g. `./venv/bin/python -m src.data_prep`.

---

## 5. Dependencies explained, package by package

### 5.1 `requirements.txt` (full development environment)

| Package | Why it is here |
|---|---|
| `pandas` | Loading CSVs, feature engineering (rolling windows, joins), parquet I/O |
| `numpy` | Numeric arrays underneath pandas/scikit-learn |
| `scikit-learn` | The model (`RandomForestClassifier`), preprocessing, metrics, and **cost-complexity pruning** (`ccp_alpha`) |
| `pyarrow` | Fast, typed Parquet read/write for `features.parquet` |
| `joblib` | Serializing the trained model + feature schema to disk |
| `fastapi` | The REST API framework for real-time inference |
| `uvicorn[standard]` | ASGI server that runs the FastAPI app |
| `pydantic` | Request/response validation and schema definition |
| `evidently` | Drift detection and data-quality reports (pinned to `<0.5`; Evidently's API changed substantially across versions) |

### 5.2 `requirements-api.txt` (lean serving container)

The Docker image only needs to **serve** the model, so it installs a reduced
set (no `evidently`, `plotly`, `statsmodels`, `nltk`, …). This keeps the image
small and the build fast. `scikit-learn` is **pinned to `1.5.2`** — the exact
version used for training — so the pickled model unpickles without
version-mismatch warnings.

---

## 6. Stage 1 — Data preparation (`src/data_prep.py`)

### 6.1 What it does

Turns the five raw CSVs into a single labeled, feature-engineered table.

1. **Rolling telemetry features** — for each of the 4 sensors, the mean and
   standard deviation over a **short (3-hour)** and a **long (24-hour)** window,
   computed **per machine** so windows never leak across units. This is the
   standard predictive-maintenance feature recipe: it captures both the recent
   operating point and its short/long-term variability.
2. **Error-count features** — for each error type, the count over the trailing
   24 hours (sparse error events aligned onto the hourly grid, then rolling-summed).
3. **Machine metadata** — static `model` (categorical) and `age` (numeric).
4. **Down-sampling** — one feature row every **3 hours** (reduces redundancy;
   consecutive hourly rows are near-duplicates).
5. **Labeling** — `failure_within_24h = 1` if that machine has a component
   failure within the next 24 hours, else `0`. This is the AMR "the unit will
   fail its task" target.

### 6.2 How to run

```bash
python -m src.data_prep
```

Run as a module (`-m`) from the project root so the `src` package imports
resolve.

### 6.3 Expected output

```
Engineering telemetry features...
Engineering error-count features...
Merging feature blocks + machine metadata...
Labeling (failure within next 24h)...

=== Done ===
Rows:            292,000
Feature columns: 23
Positives:       5,728 (1.96%)
Saved ->         .../data/processed/features.parquet
```

**Interpretation:** 292,000 feature rows, 23 features, and a **1.96% positive
rate** — failures are rare, so the classes are imbalanced. This is expected and
is handled at training time with class weighting. The output file
`data/processed/features.parquet` is the single input to training.

### 6.4 The 23 features

- 16 rolling sensor stats: `{volt,rotate,pressure,vibration}_{mean,std}_{3h,24h}`
- 5 error counts: `error{1..5}_count_24h`
- 2 metadata: `model` (categorical), `age` (numeric)

(The parquet also keeps `datetime` and `machineID` for splitting/labeling; these
are **not** model inputs.)

---

## 7. Stage 2 — Training and pruning (`src/train.py`)

### 7.1 What it does

Trains **two** models on identical data and compares them:

- **baseline** — `RandomForestClassifier`, no pruning (`ccp_alpha = 0`)
- **pruned** — same, with **cost-complexity pruning** (`ccp_alpha > 0`)

It reports the trade-off that matters for a real-time service: **model size on
disk** and **inference latency** versus **predictive quality** (PR-AUC). The
pruned model is saved as the production artifact.

### 7.2 Key design decisions

- **Time-based split, never random.** Telemetry is a time series; a random split
  would leak *future* readings into the training set and inflate the scores. We
  train on the earlier period and test on the later one (last 20% of the
  timeline). See `time_split()`.
- **Class weighting.** With only ~2% positives, `class_weight="balanced"` stops
  the model from trivially predicting "no failure" for everything.
- **PR-AUC as the headline metric.** Under heavy class imbalance, accuracy is
  meaningless (98% by always predicting "no failure"). **Average precision
  (PR-AUC)** measures how well the model ranks the rare positive class.
- **Cost-complexity pruning (`ccp_alpha`).** scikit-learn's `RandomForest`
  applies Breiman's cost-complexity pruning to **every tree** when `ccp_alpha`
  is set, removing the weakest branches. We search a small grid and keep the
  **strongest** pruning that retains ≥98% of the baseline PR-AUC — i.e. maximum
  compression with negligible quality loss.

### 7.3 How to run

```bash
python -m src.train
```

Takes roughly one to two minutes (it trains several forests).

### 7.4 Expected output

```
Train: 233,600 rows (2.00% pos) | Test: 58,400 rows (1.79% pos)
Split at: 2015-10-20 09:00:00

Training baseline (ccp_alpha=0)...
Searching pruning strength (ccp_alpha)...
  alpha=1e-05    PR-AUC=0.8172 (ok)
  alpha=5e-05    PR-AUC=0.8178 (ok)
  alpha=0.0001   PR-AUC=0.8204 (ok)
  alpha=0.0005   PR-AUC=0.8182 (ok)

=== Baseline vs Pruned ===
metric                  baseline        pruned
ccp_alpha                      0        0.0005
PR-AUC                    0.8138        0.8182
ROC-AUC                   0.9898        0.9950
tree nodes               368,490         9,214
size (MB)                  29.52          0.78
latency (ms)              15.540        15.573

Production model saved -> models/model.joblib
Feature schema saved   -> models/feature_columns.joblib
```

### 7.5 Interpreting the compression result

| Metric | Baseline | Pruned | Change |
|---|---|---|---|
| PR-AUC | 0.814 | **0.818** | quality held (even slightly up) |
| Tree nodes | 368,490 | **9,214** | **40× fewer** |
| Size on disk | 29.52 MB | **0.78 MB** | **38× smaller** |
| Single-row latency | 15.54 ms | 15.57 ms | **unchanged** |

**The honest, important nuance:** pruning delivered a huge **footprint** win
(38× smaller model, 40× fewer nodes) with **no accuracy loss**, but **latency
was flat**. This is because per-request latency here is dominated by
Python/scikit-learn dispatch overhead, not tree traversal. The correct
conclusion — and a strong point for the analysis section — is that the pruning
payoff for this model is **deployment footprint** (container size, RAM,
cold-start), not inference speed. Do not claim a latency improvement.

### 7.6 Artifacts produced

- `models/model.joblib` — the **pruned** production model (a full scikit-learn
  `Pipeline`: one-hot encoding + forest). The API loads exactly this.
- `models/model_baseline.joblib` — unpruned model, kept for the comparison.
- `models/feature_columns.joblib` — the exact feature order, so the API can
  build inputs the model expects.

---

## 8. Stage 3 — Model serving (`api/`)

### 8.1 Components

- **`api/schemas.py`** — `TelemetryFeatures` (the 23-field request body, with an
  example) and `PredictionResponse` (`failure_probability`, `failure_predicted`,
  `threshold`). Pydantic validates every request automatically.
- **`api/db.py`** — SQLite logging. On startup `init_db()` creates
  `monitoring/predictions.db` with a `predictions` table; every prediction is
  inserted with a timestamp, the input features (as JSON), the probability, and
  the 0/1 decision. This is the historical record the monitor reads.
- **`api/main.py`** — the FastAPI application. Loads the pruned model + feature
  order once at startup (via a lifespan handler) and exposes three endpoints.

### 8.2 Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + whether the model is loaded + the decision threshold |
| POST | `/predict` | Failure probability for one telemetry feature vector (and logs it) |
| GET | `/stats` | Number of logged predictions (quick monitoring sanity check) |

The decision threshold defaults to `0.5` and is configurable via the
`DECISION_THRESHOLD` environment variable.

### 8.3 How to run

```bash
uvicorn api.main:app --port 8000
# add --reload during development for auto-restart on code changes
```

Interactive API docs (Swagger UI) are then at **http://127.0.0.1:8000/docs**.

### 8.4 Example requests

Health check:

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"threshold":0.5}
```

A prediction (low-risk telemetry):

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "volt_mean_3h":170.2,"volt_std_3h":12.1,"volt_mean_24h":169.8,"volt_std_24h":14.9,
    "rotate_mean_3h":449.5,"rotate_std_3h":40.2,"rotate_mean_24h":447.1,"rotate_std_24h":48.0,
    "pressure_mean_3h":98.9,"pressure_std_3h":8.4,"pressure_mean_24h":100.1,"pressure_std_24h":10.2,
    "vibration_mean_3h":40.1,"vibration_std_3h":4.2,"vibration_mean_24h":40.4,"vibration_std_24h":5.1,
    "error1_count_24h":0,"error2_count_24h":0,"error3_count_24h":0,"error4_count_24h":0,"error5_count_24h":0,
    "model":"model3","age":18
  }'
# {"failure_probability":0.0049...,"failure_predicted":false,"threshold":0.5}
```

Verify logging:

```bash
curl -s http://127.0.0.1:8000/stats     # {"logged_predictions": N}
```

Inspect the SQLite log directly:

```bash
sqlite3 monitoring/predictions.db \
  "SELECT id, ts, round(failure_probability,4), failure_predicted FROM predictions LIMIT 5;"
```

---

## 9. Stage 4 — Monitoring and observability (`monitoring/`)

### 9.1 The idea

Monitoring compares two distributions:

- **reference** — the model's **training-period** window (what "normal" looked
  like), with the model's own predicted probabilities.
- **current** — recent **production** requests, read back from the SQLite log.

EvidentlyAI computes **data drift** (per-feature distribution shift),
**prediction drift** (shift in the model's output distribution), and **data
quality**. Two artifacts are produced: an interactive HTML report and a
machine-readable JSON verdict (the hook an automated retraining trigger would
read).

### 9.2 The two scripts

- **`monitoring/replay.py`** — samples 2,000 rows from the **post-training**
  period and POSTs them to the running API, so they are logged as production
  traffic. With `--drift`, it first applies a **synthetic fleet-wide fault**
  (elevated vibration and pressure) as a **positive control** for the monitor.
- **`monitoring/monitor.py`** — builds reference + current, runs Evidently, saves
  `reports/drift_report<label>.html` and `reports/drift_status<label>.json`, and
  prints a summary + verdict. The `--label` flag names the output files.

### 9.3 The observability rule (`decide()`)

An alert fires if **any** of the following hold:

- dataset-level drift is flagged, **or**
- ≥30% of input features drifted, **or**
- **prediction (output) drift** is detected.

Prediction drift is the most important trigger: a shifted output distribution
means the model's behaviour has changed, regardless of which inputs moved.

### 9.4 How to run — the two scenarios

The monitor reads the SQLite log, so the **API must be running** and the log
must be populated first. Run each scenario against a **fresh** log.

**Scenario A — healthy (no false alarm):**

```bash
# terminal 1: start the API
uvicorn api.main:app --port 8000

# terminal 2:
rm -f monitoring/predictions.db          # fresh log
python -m monitoring.replay              # replay real later-period traffic
python -m monitoring.monitor --label healthy
```

**Scenario B — injected drift (true positive):**

```bash
# with the API still running in terminal 1:
rm -f monitoring/predictions.db          # fresh log
python -m monitoring.replay --drift      # inject the synthetic fault
python -m monitoring.monitor --label drift
```

### 9.5 Expected results

| Scenario | Drifted columns | Prediction drift | Verdict |
|---|---|---|---|
| A: Healthy | 0 (0%) | False (score ≈ 0.04) | ✅ OK — no significant drift |
| B: Injected fault | 7 (29.2%) | **True (score ≈ 1.18)** | 🚨 DRIFT DETECTED — prediction (output) drift |

This pair is the complete monitoring story: **no false alarm** on genuine
production data, and a **correct alert** when a real fault is injected. Open the
HTML reports in a browser to see per-feature distributions; read
`reports/drift_status_*.json` for the machine-readable verdict.

Example `reports/drift_status_drift.json`:

```json
{
  "dataset_drift": false,
  "drifted_columns": 7,
  "share_drifted": 0.2916666666666667,
  "prediction_drift": { "drift_detected": true, "drift_score": 1.184... },
  "drift_alert": true,
  "reasons": ["prediction (output) drift"]
}
```

> **Note on the natural-drift result.** Real later-period data shows *no* drift
> because the Azure telemetry is fairly stationary across the year. That is a
> correct result (the detector does not raise false alarms). The injected-drift
> scenario is what demonstrates the detector actually *fires* when it should.

---

## 10. Stage 5 — Docker

> Docker was **not installed on the development machine**, so the image below
> was authored but **not built/tested locally**. The commands are standard and
> the image is self-contained (verified: the serving code imports only
> `src.config` + `api/*` + the pinned libraries in `requirements-api.txt`).

### 10.1 What the image contains

`Dockerfile` builds a lean **serving-only** image: Python 3.10-slim +
`requirements-api.txt` + `src/` + `api/` + `models/`. Monitoring/EDA code and
the raw dataset are excluded via `.dockerignore`, keeping the image small. The
container's single job is real-time inference on port 8000.

### 10.2 Install Docker (if needed, Ubuntu)

```bash
sudo apt update && sudo apt install docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER      # then log out/in so `docker` works without sudo
```

### 10.3 Build

```bash
docker build -t amr-fleet-api .
```

### 10.4 Run

```bash
docker run --rm -p 8000:8000 amr-fleet-api
```

The API is then reachable exactly as in section 8:

```bash
curl -s http://127.0.0.1:8000/health
```

### 10.5 Persisting the request log

The container writes its SQLite log to `/app/monitoring/predictions.db`, which
disappears when the container is removed. To keep the log on the host (so you
can run the monitor against real container traffic), mount a volume:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/monitoring:/app/monitoring" \
  amr-fleet-api
```

### 10.6 Configuring the decision threshold

```bash
docker run --rm -p 8000:8000 -e DECISION_THRESHOLD=0.3 amr-fleet-api
```

---

## 11. End-to-end quickstart

From a clean checkout to a fully monitored, running system:

```bash
# 0. system prerequisite (one time, Debian/Ubuntu)
sudo apt install python3.10-venv

# 1. environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. put the Kaggle CSVs in dataset/ (5 PdM_*.csv files)

# 3. build features           -> data/processed/features.parquet
python -m src.data_prep

# 4. train + prune            -> models/model.joblib
python -m src.train

# 5. serve (leave running)    -> http://127.0.0.1:8000/docs
uvicorn api.main:app --port 8000

# 6. monitor (new terminal, venv active)
#    healthy scenario
rm -f monitoring/predictions.db
python -m monitoring.replay
python -m monitoring.monitor --label healthy
#    injected-drift scenario
rm -f monitoring/predictions.db
python -m monitoring.replay --drift
python -m monitoring.monitor --label drift

# 7. (optional) containerize
docker build -t amr-fleet-api .
docker run --rm -p 8000:8000 amr-fleet-api
```

---

## 12. Design decisions and FAQ

**Why predict "failure within 24h" rather than "task success" directly?**
The dataset labels component failures, not task outcomes. A failing unit is
precisely a unit that will not complete its task, so failure-within-a-horizon is
the natural, honest proxy for "task execution success" — and predictive
maintenance *is* fleet-health monitoring, which matches the project title
tightly.

**Why is the positive rate only ~2%?**
Failures are genuinely rare. This is why we use class weighting and evaluate with
PR-AUC rather than accuracy.

**Why did pruning not improve latency?**
Per-request latency is dominated by Python/scikit-learn call overhead, not tree
traversal, so a smaller forest predicts in about the same wall-clock time. The
pruning benefit is footprint (size/RAM/cold-start), which is the honest framing.

**Why does the healthy scenario show no drift?**
The telemetry is roughly stationary across 2015, so there is little natural
drift — the correct, no-false-alarm result. The injected-drift scenario is the
positive control that proves the detector fires when it should.

**Why one-hot encode inside the model pipeline?**
The saved artifact is a full scikit-learn `Pipeline` (encoder + forest), so the
API just calls `predict_proba` on raw feature values — no preprocessing logic is
duplicated in the serving layer, eliminating train/serve skew.

---

## 13. Mapping to the theory

The theoretical part of the project concerns drift detection and AI
observability in autonomous-fleet systems. The implementation demonstrates:

- **Data drift** — per-feature distribution shift, detected by Evidently's
  `DataDriftPreset` (statistical tests per column). Corresponds to changes in the
  *input* telemetry distribution (e.g. sensors degrading fleet-wide).
- **Prediction drift** — shift in the model's *output* distribution, the earliest
  observable symptom of concept drift when ground-truth labels are delayed (as
  failure labels always are in production).
- **AI observability** — every request is logged (SQLite), and the monitor emits
  a machine-readable verdict (`drift_status_*.json`). This is the interface an
  **automated retraining trigger** would consume: `drift_alert == true` →
  schedule a retrain on recent data.
- **Automated retraining (design)** — the `decide()` rule is deliberately the
  policy layer: dataset drift, a large share of drifted features, or prediction
  drift each raise the alert. In a full system this JSON would gate a retraining
  pipeline; here it is produced and demonstrated with a positive control.

---

## 14. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ensurepip is not available` when creating the venv | `sudo apt install python3.10-venv` |
| `ModuleNotFoundError: No module named 'src'` | Run scripts as modules from the project root: `python -m src.train`, not `python src/train.py` |
| `No logged predictions found` from the monitor | Start the API and run `python -m monitoring.replay` first; the monitor reads the SQLite log |
| Monitor connection errors during replay | The API is not running — start `uvicorn api.main:app --port 8000` in another terminal |
| Model unpickling warning in Docker | Ensure `requirements-api.txt` pins the same `scikit-learn` version used for training (`1.5.2`) |
| Port 8000 already in use | Stop the old server (`pkill -f "uvicorn api.main"`) or use `--port 8001` |
```
