"""Feature extraction for telemetry events.

FEATURE_NAMES is the single source of truth for feature order; training and
serving both import it so they cannot drift apart.
"""
FEATURE_NAMES = [
    "latency_ms",
    "error_count",
    "timeout_count",
    "traffic_rpm",
    "cpu_percent",
    "memory_percent",
]

DEFAULTS = {
    "latency_ms": 0.0,
    "error_count": 0.0,
    "timeout_count": 0.0,
    "traffic_rpm": 0.0,
    # Resource metrics default to typical idle values rather than 0, because a
    # genuine 0% CPU reading is itself anomalous and a missing field should not
    # be scored as one.
    "cpu_percent": 40.0,
    "memory_percent": 55.0,
}


def extract_features(event):
    """Return the feature vector for one telemetry event, in FEATURE_NAMES order."""
    if isinstance(event, str):
        raise TypeError(
            "extract_features expects a telemetry mapping, not a raw string. "
            f"Provide keys: {', '.join(FEATURE_NAMES)}"
        )
    return [float(event.get(name, DEFAULTS[name])) for name in FEATURE_NAMES]


def features_as_dict(features):
    return dict(zip(FEATURE_NAMES, features))
