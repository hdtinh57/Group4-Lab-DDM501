"""
ML Model wrapper for movie rating prediction.

"""

import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional

from app.config import MODEL_PATH
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieRatingModel:
    """
    Wrapper class for the movie rating prediction model.
    
    This class handles:
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to the saved model file (.pkl)
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
    
    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.
        
        Args:
            user_id: User ID (string)
            movie_id: Movie ID (string)
            
        Returns:
            Predicted rating (float between 1.0 and 5.0)
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded.")
        prediction = self.model.predict(user_id, movie_id)
        return round(prediction.est, 2)
    
    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            pairs: List of (user_id, movie_id) tuples
            
        Returns:
            List of predicted ratings
        """
        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

_model_instance: Optional[MovieRatingModel] = None

def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance
