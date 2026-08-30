"""Serving-side anomaly scoring, backed by the fitted model artifact.

The previous `default_model()` returned an `AnomalyModel` built from two
hardcoded arrays -- `DEFAULT_MEAN = [150.0, 0.5, 0.2, 1000.0]` and a matching
std -- that had never been fitted to anything. A `fit()` function existed but
nothing in the serving path called it, so the README's claim of a "fitted
baseline anomaly model" was false in production.

This loads the artifact produced by `training/train.py`. If it is missing, it
raises rather than falling back to invented constants.
"""
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from pipeline.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "anomaly_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
BASELINES_PATH = ARTIFACT_DIR / "service_baselines.json"

_MISSING = (
    f"Model artifact not found at {MODEL_PATH}.\n"
    "Train it first:\n"
    "    python training/generate_timeseries.py\n"
    "    python training/train.py"
)


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the fitted detector, per-service baselines and metadata once."""
    for path in (MODEL_PATH, METRICS_PATH, BASELINES_PATH):
        if not path.exists():
            raise FileNotFoundError(_MISSING)

    import joblib
    import sklearn

    metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    baselines = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))["baselines"]

    trained_with = metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(
            f"Detector was fitted with scikit-learn {trained_with} but "
            f"{sklearn.__version__} is installed. Re-run training/train.py if "
            "scores look wrong.",
            RuntimeWarning,
            stacklevel=2,
        )

    return joblib.load(MODEL_PATH), baselines, metadata


def known_services():
    _, baselines, _ = load_artifacts()
    return sorted(baselines)


def _standardise(features, service):
    """Scale against the service's own healthy baseline.

    Services have different healthy ranges -- payments idles near 240ms, search
    near 95ms -- so a single global scale would flag one and miss the other.
    An unknown service falls back to the average baseline and the response says so.
    """
    _, baselines, _ = load_artifacts()
    known = service in baselines

    if known:
        stats = baselines[service]
    else:
        stats = {
            name: {
                "mean": float(np.mean([b[name]["mean"] for b in baselines.values()])),
                "std": float(np.mean([b[name]["std"] for b in baselines.values()])),
            }
            for name in FEATURE_NAMES
        }

    vector = [
        (float(value) - stats[name]["mean"]) / stats[name]["std"]
        for value, name in zip(features, FEATURE_NAMES)
    ]
    return np.array(vector, dtype=float), known


def score(features, service="unknown"):
    """Anomaly score for one event. Higher means more anomalous."""
    model, _, _ = load_artifacts()
    vector, _ = _standardise(features, service)
    return float(-model.score_samples(vector.reshape(1, -1))[0])


def predict(features, service="unknown"):
    """Score, alert decision, and the per-signal deviations behind it."""
    model, _, metadata = load_artifacts()
    vector, known = _standardise(features, service)

    anomaly_score = float(-model.score_samples(vector.reshape(1, -1))[0])
    threshold = float(metadata["threshold"])

    deviations = sorted(
        ((name, round(float(z), 3)) for name, z in zip(FEATURE_NAMES, vector)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    return {
        "score": round(anomaly_score, 6),
        "threshold": round(threshold, 6),
        "is_anomaly": bool(anomaly_score >= threshold),
        "service_baseline_known": known,
        "deviations": deviations,
    }


def is_anomaly(features, service="unknown"):
    return predict(features, service)["is_anomaly"]


def model_metadata():
    """Metadata for the /health surface."""
    _, _, metadata = load_artifacts()
    return {
        "model_type": metadata["model_type"],
        "data_source": metadata["data_source"],
        "split": metadata["split"],
        "threshold": metadata["threshold"],
        "target_precision": metadata["target_precision"],
        "test_f1": metadata["test"]["f1"],
        "test_episode_recall": metadata["test"]["episode_recall"],
        "sklearn_version": metadata["sklearn_version"],
    }
