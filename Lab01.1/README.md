# Lab 01.1 — Movie Rating Prediction API

A containerized FastAPI service that serves SVD-based movie rating predictions
trained on the MovieLens 100K dataset.

## Included deliverables

- Pre-trained Surprise SVD model at `models/svd_model.pkl`.
- FastAPI endpoints: `/health`, `/predict`, `/predict/batch`, and `/model/info`.
- Input/output validation with Pydantic schemas.
- Docker image and Compose configuration.
- Automated API test suite.

## Run

```bash
docker compose up --build
```

The API is then available at `http://localhost:8000`; its interactive OpenAPI
documentation is available at `http://localhost:8000/docs`.

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id":"196","movie_id":"242"}'
```

Example response:

```json
{
  "user_id": "196",
  "movie_id": "242",
  "predicted_rating": 3.72
}
```

## Test

```bash
pytest -q --cov=app --cov-report=term-missing
```

The completed suite contains 16 API tests.
