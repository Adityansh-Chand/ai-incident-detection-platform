
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.metrics import metrics
from models.anomaly_model import known_services, model_metadata, predict
from pipeline.features import extract_features, features_as_dict
from utils.security import request_id_middleware, require_api_key
from utils.storage import recent_events, save_event

app = FastAPI(title="AI Incident Detection Platform", version="1.0.0")
app.middleware("http")(request_id_middleware)


class IncidentEvent(BaseModel):
    service: str = "unknown"
    latency_ms: float = Field(..., ge=0)
    error_count: int = Field(0, ge=0)
    timeout_count: int = Field(0, ge=0)
    traffic_rpm: float = Field(0, ge=0)
    # Present in datasets/production_schema.json and used by the fitted model.
    # Optional so existing four-field callers keep working; omitted values fall
    # back to typical idle levels rather than zero.
    cpu_percent: float = Field(40.0, ge=0, le=100)
    memory_percent: float = Field(55.0, ge=0, le=100)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    metrics.increment("http_errors_total")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": str(request.url.path)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    metrics.increment("validation_errors_total")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request",
            "details": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    metrics.increment("unhandled_errors_total")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "path": str(request.url.path)},
    )


@app.get("/")
def health():
    return {"status": "running"}


@app.get("/health")
def health_check():
    """Health plus the identity of the detector actually loaded."""
    return {
        "status": "running",
        "model": model_metadata(),
        "services_with_fitted_baselines": known_services(),
    }


@app.get("/metrics")
def metrics_endpoint():
    return metrics.snapshot()


@app.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20):
    return {"events": recent_events(limit=min(limit, 100))}


@app.post("/score", dependencies=[Depends(require_api_key)])
def score_event(event: IncidentEvent):
    metrics.increment("scores_total")
    features = extract_features(event.model_dump())
    prediction = predict(features, event.service)
    result = {
        "service": event.service,
        "features": features_as_dict(features),
        **prediction,
    }
    save_event("incident_score", result)
    return result
