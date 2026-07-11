# AMR Fleet Failure-Prediction API
# Containerizes the FastAPI inference service (api/) together with the trained,
# pruned model artifact (models/model.joblib). Monitoring/EDA code is NOT copied
# in: the container's single job is real-time inference.

FROM python:3.10-slim

# Don't buffer stdout/stderr (so logs stream) and don't write .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install serving dependencies first so this layer is cached across code changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# Application code + the pruned production model.
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# The API writes request logs to /app/monitoring/predictions.db at runtime;
# db.init_db() creates this directory on startup. Mount a volume here to persist
# logs across container restarts (see docs/PROJECT_GUIDE.md).

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
