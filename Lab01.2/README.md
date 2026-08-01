# Lab 01.2 — MLOps Pipeline and Experiment Tracking

An end-to-end recommendation-model workflow for MovieLens 100K. The project
trains Surprise models, records experiments in MLflow, registers the best model,
and exposes the workflow through an Airflow DAG.

## Included deliverables

- Data ingestion, validation, preprocessing, training, and evaluation stages.
- SVD, NMF, and KNN model configurations.
- MLflow parameter, metric, artifact, and `pyfunc` model logging.
- Model selection and MLflow Model Registry promotion.
- Airflow DAG: `movie_rating_training`.
- Docker Compose stack for MLflow, PostgreSQL, and Airflow.
- Experiment report with nine completed tuning runs and MLflow UI evidence.

## Run the Docker stack

```bash
docker compose up --build
```

Services:

- MLflow: `http://localhost:5000`
- Airflow: `http://localhost:8080` (username/password: `admin` / `admin`)

## Run the pipeline locally

```bash
python -m pipeline.run_pipeline --model-type svd --n-factors 100 --n-epochs 20
python -m experiments.run_experiments
```

## Run the pipeline in the Docker stack

After `docker compose up --build`, run the pipeline from the Airflow scheduler
service so it automatically uses the Docker MLflow server:

```bash
docker compose exec airflow-scheduler python -m pipeline.run_pipeline --model-type svd --n-factors 100 --n-epochs 20
```

The experiment results are documented in
[`docs/experiment_report.md`](docs/experiment_report.md).

## Test

```bash
pytest -q -m "not slow"
```

The completed suite covers the pipeline stages, MLflow workflow, registry flow,
and experiment reporting.
