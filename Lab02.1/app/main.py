from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import API_DESCRIPTION, API_TITLE, API_VERSION, MODEL_VERSION
from app.model import MovieRatingModel
from app.schemas import BatchPredictionRequest, BatchPredictionResponse, HealthResponse, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = MovieRatingModel()
    yield


app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = app.state.model.is_loaded()
    return HealthResponse(status="healthy" if loaded else "unhealthy", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    rating = app.state.model.predict(request.user_id, request.movie_id)
    return PredictionResponse(**request.model_dump(), predicted_rating=rating, model_version=MODEL_VERSION)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = [predict(item) for item in request.predictions]
    return BatchPredictionResponse(predictions=predictions, total_count=len(predictions))


@app.get("/")
def root() -> dict:
    return {"name": API_TITLE, "health": "/health", "docs": "/docs"}
