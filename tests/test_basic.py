from datasets.generate_timeseries import generate
from models.anomaly_model import fit, is_anomaly
from pipeline.features import extract_features


def test_high_error_event_is_anomalous():
    event = {
        "latency_ms": 900,
        "error_count": 20,
        "timeout_count": 6,
        "traffic_rpm": 2300,
    }

    assert is_anomaly(extract_features(event))


def test_fitted_model_scores_normal_events_below_threshold():
    baseline = [
        extract_features({"latency_ms": 120, "error_count": 0, "timeout_count": 0, "traffic_rpm": 950}),
        extract_features({"latency_ms": 150, "error_count": 1, "timeout_count": 0, "traffic_rpm": 990}),
        extract_features({"latency_ms": 170, "error_count": 0, "timeout_count": 1, "traffic_rpm": 1020}),
    ]
    model = fit(baseline)

    prediction = model.predict(extract_features({"latency_ms": 155, "error_count": 0, "timeout_count": 0, "traffic_rpm": 1000}))

    assert prediction["is_anomaly"] is False


def test_generator_includes_labeled_anomalies():
    rows = generate(points=60, anomaly_every=30)

    assert any(row["label"] == 1 for row in rows)
