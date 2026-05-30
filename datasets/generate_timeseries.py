
import random

def generate(points=200, anomaly_every=50, seed=7):
    random.seed(seed)
    rows = []

    for index in range(points):
        is_anomaly = index > 0 and index % anomaly_every == 0
        rows.append({
            "timestamp": index,
            "latency_ms": random.gauss(850, 90) if is_anomaly else random.gauss(145, 25),
            "error_count": random.randint(12, 25) if is_anomaly else random.randint(0, 2),
            "timeout_count": random.randint(4, 10) if is_anomaly else random.randint(0, 1),
            "traffic_rpm": random.gauss(2200, 250) if is_anomaly else random.gauss(950, 120),
            "label": int(is_anomaly),
        })

    return rows
