
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.metrics import metrics
from models.anomaly_model import known_services, model_metadata, predict
from monitoring.incident_state import (
    MIN_ANOMALIES,
    WINDOW_SECONDS,
    active_incident,
    active_incidents,
    record,
)
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
        "active_incidents": len(active_incidents()),
        "incident_state": (
            "in-memory, per-process rolling window -- a demonstration surface, "
            "not a durable incident store"
        ),
    }


@app.get("/metrics")
def metrics_endpoint():
    return metrics.snapshot()


@app.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20):
    return {"events": recent_events(limit=min(limit, 100))}


@app.get("/incidents/active", dependencies=[Depends(require_api_key)])
def list_active_incidents(service: str | None = None):
    """Which services are currently in an incident.

    Consumed by the customer operations service: a complaint about a service
    that is currently degraded is an incident symptom, not an individual issue,
    and should be handled differently.
    """
    metrics.increment("incident_lookups_total")
    if service:
        incident = active_incident(service)
        return {
            "service": service,
            "active": incident is not None,
            "incident": incident,
            "rule": f"{MIN_ANOMALIES}+ anomalous minutes within {WINDOW_SECONDS:.0f}s",
        }
    incidents = active_incidents()
    return {
        "active_count": len(incidents),
        "incidents": incidents,
        "rule": f"{MIN_ANOMALIES}+ anomalous minutes within {WINDOW_SECONDS:.0f}s",
    }


@app.post("/score", dependencies=[Depends(require_api_key)])
def score_event(event: IncidentEvent):
    metrics.increment("scores_total")
    features = extract_features(event.model_dump())
    prediction = predict(features, event.service)

    # Feed the rolling incident window so other services can ask whether this
    # service is currently degraded, rather than only scoring one minute.
    record(event.service, prediction["score"], prediction["is_anomaly"])

    result = {
        "service": event.service,
        "features": features_as_dict(features),
        **prediction,
    }
    save_event("incident_score", result)
    return result
