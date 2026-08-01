# Group 4 — DDM501 Labs

This repository contains four connected DDM501 deliverables. The movie-rating
system progresses from an API and MLOps pipeline to automated quality gates and
production observability.

## Repository layout

- `Lab01.1/` — Movie Rating Prediction API, tests, Docker image, and trained SVD model.
- `Lab01.2/` — Reproducible training pipeline, MLflow experiment tracking, model registry, and Airflow DAG.
- `Lab02.1/` — ML testing pyramid, coverage gate, linting, and GitHub Actions container validation.
- `Lab02.2/` — Prometheus metrics, alert rules, Grafana dashboards, and concurrent load testing.

## Quick start

### Lab 1: serving API

```bash
cd Lab01.1
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for Swagger documentation. The service reads
the checked-in SVD model by default; set `MODEL_PATH` to serve a promoted model
from Lab 2.

### Lab 2: training and MLOps

```bash
cd Lab01.2
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
docker compose up -d
python -m experiments.run_experiments
```

MLflow is available at `http://localhost:5000`, and Airflow is available at
`http://localhost:8080` (username/password: `admin` / `admin`). The experiment
report is written to `docs/experiment_report.md`.

## Model promotion flow

1. Lab 2 logs every training run's parameters, metrics, raw pickle, and an
   MLflow pyfunc model artifact.
2. The run with the lowest RMSE is registered as `movie-rating-model` and
   promoted to `Production`.
3. Export or download the promoted model to `Lab01.1/models/svd_model.pkl`, or
   point Lab 1's `MODEL_PATH` environment variable at that exported artifact.

This keeps training and low-latency serving independently runnable while
preserving a clear, versioned handoff between them.

## Labs 3 and 4

Lab 3 runs its test suite with an 80% coverage gate and validates the Docker
image in GitHub Actions. Lab 4 starts the complete monitoring stack with:

```bash
cd Lab02.2
docker compose up --build -d
python scripts/load_test.py --duration 30 --workers 10
```

Open Grafana at `http://localhost:3000` with `admin` / `admin`; API, Prometheus,
and Grafana run on ports 8001, 9090, and 3000 respectively.
