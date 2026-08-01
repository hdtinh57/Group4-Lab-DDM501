from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from app.config import API_DESCRIPTION, API_TITLE, API_VERSION, MODEL_VERSION
from app.metrics import PREDICTION_ERRORS, count_implemented_metrics
from app.middleware import MetricsMiddleware
from app.model import MovieRatingModel
from app.schemas import BatchPredictionRequest, BatchPredictionResponse, HealthResponse, MetricsInfo, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = MovieRatingModel()
    yield


app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION, lifespan=lifespan)
app.add_middleware(MetricsMiddleware)


@app.get("/")
def root() -> dict:
    return {"name": API_TITLE, "health": "/health", "metrics": "/metrics", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = app.state.model.is_loaded()
    return HealthResponse(status="healthy" if loaded else "unhealthy", model_loaded=loaded, model_version=MODEL_VERSION)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/info", response_model=MetricsInfo)
def metrics_info() -> MetricsInfo:
    implemented, _ = count_implemented_metrics()
    return MetricsInfo(metrics_enabled=True, endpoint="/metrics", metrics_count=implemented)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        rating, latency_ms = app.state.model.predict_with_latency(request.user_id, request.movie_id)
        return PredictionResponse(**request.model_dump(), predicted_rating=rating, model_version=MODEL_VERSION, latency_ms=round(latency_ms, 4))
    except Exception as error:
        PREDICTION_ERRORS.labels("model_error", MODEL_VERSION).inc()
        raise HTTPException(status_code=500, detail="Prediction failed") from error


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    results = [predict(item) for item in request.predictions]
    return BatchPredictionResponse(predictions=results, total_count=len(results), avg_latency_ms=round(sum(item.latency_ms for item in results) / len(results), 4))
