from typing import List
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=50)
    movie_id: str = Field(min_length=1, max_length=50)

    @field_validator("user_id", "movie_id")
    @classmethod
    def normalize(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ID cannot be blank")
        return value


class PredictionResponse(PredictionRequest):
    predicted_rating: float = Field(ge=1, le=5)
    model_version: str
    latency_ms: float = Field(ge=0)


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    avg_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class MetricsInfo(BaseModel):
    metrics_enabled: bool
    endpoint: str
    metrics_count: int
