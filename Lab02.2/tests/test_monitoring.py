from fastapi.testclient import TestClient
from app.main import app


def test_metrics_endpoint_exposes_application_and_ml_metrics():
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "ml_model_loaded" in response.text


def test_prediction_records_ml_metrics():
    with TestClient(app) as client:
        response = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
        metrics = client.get("/metrics")
    assert response.status_code == 200
    assert "ml_predictions_total" in metrics.text
    assert "ml_prediction_duration_seconds" in metrics.text


def test_monitoring_metadata_reports_all_metrics():
    with TestClient(app) as client:
        response = client.get("/metrics/info")
    assert response.json()["metrics_count"] == 8
