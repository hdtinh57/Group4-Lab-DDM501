import hashlib
import time
from app.config import MODEL_VERSION
from app.metrics import MODEL_INFO, MODEL_LAST_RELOAD, MODEL_LOADED, PREDICTION_COUNT, PREDICTION_LATENCY, PREDICTION_VALUE


class MovieRatingModel:
    """Deterministic model adapter used when a serialized SVD is not mounted."""
    def __init__(self):
        self.loaded_at = time.time()
        MODEL_LOADED.set(1)
        MODEL_LAST_RELOAD.set(self.loaded_at)
        MODEL_INFO.info({"version": MODEL_VERSION, "type": "collaborative-filtering"})

    def is_loaded(self) -> bool:
        return True

    def predict_with_latency(self, user_id: str, movie_id: str) -> tuple[float, float]:
        started = time.perf_counter()
        value = int.from_bytes(hashlib.blake2b(f"{user_id}:{movie_id}".encode(), digest_size=2).digest(), "big")
        rating = round(2.0 + (value % 301) / 100, 2)
        duration = time.perf_counter() - started
        PREDICTION_COUNT.labels(MODEL_VERSION).inc()
        PREDICTION_LATENCY.labels(MODEL_VERSION).observe(duration)
        PREDICTION_VALUE.labels(MODEL_VERSION).observe(rating)
        return rating, duration * 1000
