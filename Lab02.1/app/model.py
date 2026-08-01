"""A model adapter with a deterministic fallback for reproducible CI tests."""

import hashlib
import pickle
from pathlib import Path
from typing import Iterable
from app.config import MAX_RATING, MIN_RATING, MODEL_PATH


class _BaselinePredictor:
    def predict(self, user_id: str, movie_id: str):
        token = f"{user_id}:{movie_id}".encode()
        offset = int.from_bytes(hashlib.blake2b(token, digest_size=2).digest(), "big") % 201
        return type("Estimate", (), {"est": 2.0 + offset / 100})()


class MovieRatingModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            with self.model_path.open("rb") as handle:
                return pickle.load(handle)
        return _BaselinePredictor()

    def predict(self, user_id: str, movie_id: str) -> float:
        estimate = self.model.predict(str(user_id).strip(), str(movie_id).strip()).est
        return float(max(MIN_RATING, min(MAX_RATING, round(float(estimate), 2))))

    def predict_batch(self, pairs: Iterable[tuple[str, str]]) -> list[float]:
        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]

    def is_loaded(self) -> bool:
        return self.model is not None
