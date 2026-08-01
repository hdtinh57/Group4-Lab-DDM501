"""
Pydantic schemas for request/response validation.

"""

from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    user_id: str = Field(..., description="The ID of the user", json_schema_extra={"example": "196"})
    movie_id: str = Field(..., description="The ID of the movie", json_schema_extra={"example": "242"})

class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    model_config = {
        "protected_namespaces": ()
    }
    user_id: str = Field(..., description="The ID of the user")
    movie_id: str = Field(..., description="The ID of the movie")
    predicted_rating: float = Field(..., description="The predicted rating (between 1.0 and 5.0)", ge=1.0, le=5.0)
    model_version: str = Field(..., description="The version of the prediction model used")

class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    model_config = {
        "protected_namespaces": ()
    }
    status: str = Field(..., description="Status of the API service, e.g. 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Flag indicating whether the ML model is successfully loaded")

class PredictionItem(BaseModel):
    """Single prediction item for batch requests."""
    user_id: str
    movie_id: str


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    predictions: List[PredictionItem]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    predictions: List[PredictionResponse]
    total_count: int
