FEATURE_NAMES = ["latency_ms", "error_count", "timeout_count", "traffic_rpm"]


def extract_features(event):
    if isinstance(event, str):
        lower = event.lower()
        return [
            float(len(event)),
            float(lower.count("error")),
            float(lower.count("timeout")),
            0.0,
        ]

    return [
        float(event.get("latency_ms", 0)),
        float(event.get("error_count", 0)),
        float(event.get("timeout_count", 0)),
        float(event.get("traffic_rpm", 0)),
    ]


def features_as_dict(features):
    return dict(zip(FEATURE_NAMES, features))
