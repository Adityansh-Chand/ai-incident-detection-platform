
from fastapi import FastAPI
from pydantic import BaseModel, Field

from models.anomaly_model import default_model
from pipeline.features import extract_features, features_as_dict

app = FastAPI()
model = default_model()


class IncidentEvent(BaseModel):
    service: str = "unknown"
    latency_ms: float = Field(..., ge=0)
    error_count: int = Field(0, ge=0)
    timeout_count: int = Field(0, ge=0)
    traffic_rpm: float = Field(0, ge=0)

@app.get("/")
def health():
    return {"status": "running"}


@app.get("/health")
def health_check():
    return {"status": "running", "threshold": model.threshold}


@app.post("/score")
def score_event(event: IncidentEvent):
    features = extract_features(event.model_dump())
    prediction = model.predict(features)
    return {
        "service": event.service,
        "features": features_as_dict(features),
        **prediction,
    }
