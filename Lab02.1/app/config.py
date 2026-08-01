from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "svd_model.pkl"
MODEL_VERSION = "2.0.0"
API_TITLE = "Movie Rating Prediction API"
API_DESCRIPTION = "Tested ML prediction service for DDM501 Lab 3"
API_VERSION = "2.0.0"
MIN_RATING, MAX_RATING = 1.0, 5.0
