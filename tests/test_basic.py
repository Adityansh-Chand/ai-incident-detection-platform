"""Behavioural tests for the fitted anomaly detector."""
import pytest

from models.anomaly_model import is_anomaly, known_services, load_artifacts, predict, score
from pipeline.features import FEATURE_NAMES, extract_features, features_as_dict
from training.generate_timeseries import generate

HEALTHY = {
    "latency_ms": 180, "error_count": 0, "timeout_count": 0,
    "traffic_rpm": 1400, "cpu_percent": 46, "memory_percent": 58,
}
DEGRADED = {
    "latency_ms": 620, "error_count": 18, "timeout_count": 6,
    "traffic_rpm": 1150, "cpu_percent": 78, "memory_percent": 70,
}
# Load rises sharply but health does not degrade. The detector must NOT alert:
# this is the case a naive "unusual traffic" rule gets wrong.
BENIGN_SPIKE = {
    "latency_ms": 205, "error_count": 0, "timeout_count": 0,
    "traffic_rpm": 2600, "cpu_percent": 55, "memory_percent": 60,
}


def test_degraded_event_is_flagged():
    assert is_anomaly(extract_features(DEGRADED), "checkout")


def test_healthy_event_is_not_flagged():
    assert not is_anomaly(extract_features(HEALTHY), "checkout")


def test_benign_traffic_spike_is_not_flagged():
    """Rising load alone is not an incident."""
    assert not is_anomaly(extract_features(BENIGN_SPIKE), "checkout")


def test_degraded_scores_above_healthy():
    assert score(extract_features(DEGRADED), "checkout") > score(
        extract_features(HEALTHY), "checkout"
    )


def test_prediction_reports_threshold_and_deviations():
    result = predict(extract_features(DEGRADED), "checkout")
    assert result["is_anomaly"] is True
    assert result["score"] >= result["threshold"]
    assert {name for name, _ in result["deviations"]} == set(FEATURE_NAMES)

    magnitudes = [abs(value) for _, value in result["deviations"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_error_count_dominates_for_a_clear_incident():
    result = predict(extract_features(DEGRADED), "checkout")
    assert result["deviations"][0][0] in {"error_count", "timeout_count", "latency_ms"}


def test_unknown_service_is_scored_but_flagged_as_such():
    """Falls back to an averaged baseline and says so, rather than failing."""
    result = predict(extract_features(DEGRADED), "not-a-real-service")
    assert result["service_baseline_known"] is False
    assert predict(extract_features(DEGRADED), "checkout")["service_baseline_known"] is True


def test_per_service_baselines_were_fitted():
    assert set(known_services()) == {"checkout", "payments", "search", "identity"}


def test_service_baselines_actually_differ():
    """If they were identical, per-service standardisation would be pointless."""
    _, baselines, _ = load_artifacts()
    latencies = {s: b["latency_ms"]["mean"] for s, b in baselines.items()}
    assert max(latencies.values()) > 2 * min(latencies.values())


def test_extract_features_rejects_raw_strings():
    """The old version silently scored len(text) as latency."""
    with pytest.raises(TypeError):
        extract_features("some log line")


def test_missing_resource_fields_default_to_idle_not_zero():
    """A genuine 0% CPU reading is anomalous; a missing field is not."""
    features = features_as_dict(extract_features({"latency_ms": 180}))
    assert features["cpu_percent"] > 0
    assert features["memory_percent"] > 0


def test_generator_produces_multi_minute_episodes():
    frame = generate()
    episodes = frame[frame.incident_id != ""].groupby("incident_id").size()
    assert len(episodes) > 20
    assert episodes.min() >= 5, "incidents must span minutes, not single points"


def test_generator_anomaly_rate_stays_rare():
    frame = generate()
    assert 0.01 < frame.label.mean() < 0.08
