from fastapi.testclient import TestClient
from app.main import app
from app.schemas import PredictionRequest


def test_prediction_request_strips_valid_ids():
    request = PredictionRequest(user_id=" 196 ", movie_id=" 242 ")
    assert (request.user_id, request.movie_id) == ("196", "242")


def test_prediction_request_rejects_blank_identifier():
    from pydantic import ValidationError

    try:
        PredictionRequest(user_id="   ", movie_id="242")
    except ValidationError:
        return
    raise AssertionError("blank identifiers must be rejected")


def test_api_health_and_prediction_are_available():
    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json={"user_id": "196", "movie_id": "242"})
    assert health.json()["status"] == "healthy"
    assert 1.0 <= prediction.json()["predicted_rating"] <= 5.0


def test_invalid_api_payload_returns_validation_error():
    with TestClient(app) as client:
        response = client.post("/predict", json={"user_id": "", "movie_id": "242"})
    assert response.status_code == 422


def test_model_prediction_is_invariant_to_whitespace():
    with TestClient(app) as client:
        first = client.post("/predict", json={"user_id": "196", "movie_id": "242"}).json()
        second = client.post("/predict", json={"user_id": " 196 ", "movie_id": " 242 "}).json()
    assert first["predicted_rating"] == second["predicted_rating"]
