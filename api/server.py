
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.metrics import metrics
from models.anomaly_model import default_model
from pipeline.features import extract_features, features_as_dict
from utils.security import request_id_middleware, require_api_key
from utils.storage import recent_events, save_event

app = FastAPI(title="AI Incident Detection Platform", version="1.0.0")
app.middleware("http")(request_id_middleware)
model = default_model()


class IncidentEvent(BaseModel):
    service: str = "unknown"
    latency_ms: float = Field(..., ge=0)
    error_count: int = Field(0, ge=0)
    timeout_count: int = Field(0, ge=0)
    traffic_rpm: float = Field(0, ge=0)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    metrics.increment("http_errors_total")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": str(request.url.path)},
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
    return {"status": "running", "threshold": model.threshold}


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
    prediction = model.predict(features)
    result = {
        "service": event.service,
        "features": features_as_dict(features),
        **prediction,
    }
    save_event("incident_score", result)
    return result
