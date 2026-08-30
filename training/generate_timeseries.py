"""Generate the synthetic telemetry series used to fit the anomaly detector.

The previous generator marked a point anomalous when `index % anomaly_every == 0`
and drew its values from a clearly separated distribution. That is not an anomaly
detection problem: a single threshold on error_count solves it perfectly, and any
metric computed on it is meaningless.

This generates something a detector can actually be wrong about:

- **Incidents are episodes, not points.** Each spans consecutive minutes with a
  ramp, a peak and a recovery, so partial detection is possible and detection
  latency is a real quantity.
- **Signals are correlated during an incident** -- latency, errors, timeouts and
  CPU move together, the way they do when a real dependency degrades.
- **Benign traffic spikes are included as hard negatives.** Traffic rises sharply
  while latency and errors stay healthy. A detector keying on "unusual traffic"
  will fire on these and be punished for it in precision.
- **Per-service baselines differ**, so a single global threshold is not optimal.
- Anomalous points are ~2-4% of the series, which is why the evaluation reports
  precision/recall and PR-AUC rather than accuracy.

Deterministic: fixed seed, fixed start timestamp, no wall-clock reads.

    python training/generate_timeseries.py            # write datasets/telemetry.csv
    python training/generate_timeseries.py --check    # fail if output would differ
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE: lives in training/ rather than datasets/ so the package name cannot
# shadow the third-party `datasets` library.
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "datasets" / "telemetry.csv"

SEED = 20260830
START = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)
POINTS_PER_SERVICE = 10080  # seven days at one-minute resolution
INTERVAL = timedelta(minutes=1)

SERVICES = {
    "checkout":  {"latency": 180.0, "traffic": 1400.0, "cpu": 46.0, "memory": 58.0},
    "payments":  {"latency": 240.0, "traffic": 900.0, "cpu": 52.0, "memory": 63.0},
    "search":    {"latency": 95.0, "traffic": 2600.0, "cpu": 38.0, "memory": 49.0},
    "identity":  {"latency": 130.0, "traffic": 700.0, "cpu": 33.0, "memory": 44.0},
}
REGIONS = ["us-east-1", "eu-west-1"]

INCIDENTS_PER_SERVICE = 26
BENIGN_SPIKES_PER_SERVICE = 40
MIN_DURATION, MAX_DURATION = 6, 26
SEVERITIES = ["low", "medium", "high", "critical"]


def _episode_shape(length):
    """Ramp up, plateau, recover. Peak intensity 1.0 in the middle."""
    positions = np.linspace(0.0, 1.0, length)
    return np.sin(np.pi * positions) ** 0.65


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for service, base in SERVICES.items():
        region = REGIONS[len(rows) % len(REGIONS)]
        n = POINTS_PER_SERVICE

        # Diurnal traffic plus noise -- the healthy baseline.
        minutes = np.arange(n)
        daily = np.sin(2 * np.pi * minutes / 1440.0)
        traffic = base["traffic"] * (1.0 + 0.28 * daily) + rng.normal(0, base["traffic"] * 0.05, n)
        latency = base["latency"] * (1.0 + 0.10 * daily) + rng.normal(0, base["latency"] * 0.08, n)
        cpu = base["cpu"] * (1.0 + 0.18 * daily) + rng.normal(0, 3.0, n)
        memory = base["memory"] + rng.normal(0, 2.0, n) + 2.0 * daily
        errors = rng.poisson(0.35, n).astype(float)
        timeouts = rng.poisson(0.12, n).astype(float)

        label = np.zeros(n, dtype=int)
        severity = np.array([""] * n, dtype=object)
        incident_id = np.array([""] * n, dtype=object)

        # Benign traffic spikes: load rises, health does not degrade. These are
        # the hard negatives -- a naive "unusual traffic" rule fires on them.
        for _ in range(BENIGN_SPIKES_PER_SERVICE):
            length = int(rng.integers(5, 18))
            start = int(rng.integers(0, n - length))
            shape = _episode_shape(length)
            traffic[start:start + length] += base["traffic"] * 0.85 * shape
            cpu[start:start + length] += 9.0 * shape
            latency[start:start + length] += base["latency"] * 0.12 * shape

        # Real incidents: correlated degradation across signals.
        for index in range(INCIDENTS_PER_SERVICE):
            length = int(rng.integers(MIN_DURATION, MAX_DURATION))
            start = int(rng.integers(0, n - length))
            if label[max(0, start - 30):start + length + 30].any():
                continue  # keep episodes separated

            shape = _episode_shape(length)
            intensity = float(rng.uniform(0.55, 1.0))

            latency[start:start + length] += base["latency"] * (1.9 * intensity) * shape
            errors[start:start + length] += rng.poisson(11 * intensity, length) * shape
            timeouts[start:start + length] += rng.poisson(4.0 * intensity, length) * shape
            cpu[start:start + length] += 26.0 * intensity * shape
            memory[start:start + length] += 9.0 * intensity * shape
            # Traffic often DROPS during an incident as requests fail fast --
            # the opposite of a benign spike, and the distinction a detector
            # keying only on load cannot make.
            traffic[start:start + length] -= base["traffic"] * 0.18 * intensity * shape

            label[start:start + length] = 1
            # Spread the 0.55-1.0 intensity range across all four tiers.
            tier = SEVERITIES[min(int((intensity - 0.55) / 0.45 * 4), 3)]
            severity[start:start + length] = tier
            incident_id[start:start + length] = f"inc_{service}_{index:02d}"

        rows.append(
            pd.DataFrame({
                "event_id": [f"{service}_{i:05d}" for i in range(n)],
                "service": service,
                "environment": "production",
                "region": region,
                "observed_at": [(START + i * INTERVAL).isoformat() for i in range(n)],
                "latency_ms": np.clip(latency, 1, None).round(2),
                "error_count": np.clip(errors, 0, None).astype(int),
                "timeout_count": np.clip(timeouts, 0, None).astype(int),
                "traffic_rpm": np.clip(traffic, 0, None).round(2),
                "cpu_percent": np.clip(cpu, 0, 100).round(2),
                "memory_percent": np.clip(memory, 0, 100).round(2),
                "label": label,
                "severity": severity,
                "incident_id": incident_id,
            })
        )

    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    frame = generate()
    text = frame.to_csv(index=False, lineterminator="\n")

    if args.check:
        if not OUT_PATH.exists():
            print(f"FAIL: {OUT_PATH} is missing; run without --check first")
            return 1
        if OUT_PATH.read_text(encoding="utf-8") != text:
            print("FAIL: regenerated telemetry differs from the committed file")
            return 1
        print(f"OK: {OUT_PATH.name} is reproducible ({len(frame)} rows)")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    rate = frame["label"].mean()
    episodes = frame.loc[frame.incident_id != "", "incident_id"].nunique()
    print(f"wrote {OUT_PATH}")
    print(f"  rows={len(frame)} services={frame.service.nunique()} episodes={episodes}")
    print(f"  anomalous points={int(frame.label.sum())} ({rate:.3%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
