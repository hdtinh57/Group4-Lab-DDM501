from typing import List
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=50)
    movie_id: str = Field(min_length=1, max_length=50)

    @field_validator("user_id", "movie_id")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ID cannot be blank")
        return value


class PredictionResponse(BaseModel):
    user_id: str
    movie_id: str
    predicted_rating: float = Field(ge=1.0, le=5.0)
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
