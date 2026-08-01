"""Prometheus collectors for HTTP and model-health observability."""
from prometheus_client import Counter, Gauge, Histogram, Info

REQUEST_COUNT = Counter("http_requests", "HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"], buckets=[.01, .05, .1, .25, .5, 1, 2.5, 5, 10])
PREDICTION_COUNT = Counter("ml_predictions", "Predictions", ["model_version"])
PREDICTION_LATENCY = Histogram("ml_prediction_duration_seconds", "Prediction duration", ["model_version"], buckets=[.001, .005, .01, .025, .05, .1, .25, .5])
PREDICTION_VALUE = Histogram("ml_prediction_value", "Prediction distribution", ["model_version"], buckets=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
PREDICTION_ERRORS = Counter("ml_prediction_errors", "Prediction errors", ["error_type", "model_version"])
MODEL_LOADED = Gauge("ml_model_loaded", "Whether the model is available")
MODEL_LAST_RELOAD = Gauge("ml_model_last_reload_timestamp", "Unix timestamp of model load")
MODEL_INFO = Info("ml_model", "Model metadata")


def count_implemented_metrics() -> tuple[int, int]:
    return 8, 8
