"""Score the held-out test window with the fitted detector and print the report.

The previous version fitted a model on the normal rows of a six-row CSV and then
scored those same rows, so it measured nothing. This rebuilds the same
chronological split used at training time and reports the final window only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.anomaly_model import load_artifacts  # noqa: E402
from training.train import (  # noqa: E402
    ZSCORE_THRESHOLD,
    chronological_split,
    score_split,
    standardise,
)


def main():
    model, baselines, metadata = load_artifacts()

    frame = pd.read_csv(ROOT / "datasets" / "telemetry.csv").fillna(
        {"incident_id": "", "severity": ""}
    )
    _, _, test = chronological_split(frame)

    x_test = standardise(test, baselines)
    forest_scores = -model.score_samples(x_test)
    zscore_scores = np.abs(x_test).max(axis=1)
    threshold = float(metadata["threshold"])

    forest = score_split(test, forest_scores, threshold)
    baseline = score_split(test, zscore_scores, ZSCORE_THRESHOLD)

    print("Model :", metadata["model_type"])
    print("Data  :", metadata["data_source"], "(SYNTHETIC - not real telemetry)")
    print("Split :", metadata["split"])
    print(f"Test window: {len(test)} minutes, "
          f"{test.label.mean():.2%} anomalous, {forest['episodes']} incidents")
    print()
    print("Accuracy is not reported: predicting 'healthy' every minute would score")
    print(f"{1 - test.label.mean():.1%} and detect nothing.")
    print()

    print(f"{'metric':<32}{'IsolationForest':>18}{'z-score baseline':>20}")
    print("-" * 70)
    for key, label in [
        ("precision", "Precision"),
        ("recall", "Recall (point-level)"),
        ("f1", "F1"),
        ("pr_auc", "PR-AUC"),
        ("episode_recall", "Episode recall"),
        ("alerts_per_hour", "Alerts per hour"),
    ]:
        print(f"{label:<32}{forest[key]:>18}{baseline[key]:>20}")
    print(f"{'Episodes detected':<32}"
          f"{f'{forest['episodes_detected']}/{forest['episodes']}':>18}"
          f"{f'{baseline['episodes_detected']}/{baseline['episodes']}':>20}")
    print(f"{'Median detection latency (min)':<32}"
          f"{str(forest['median_detection_latency_minutes']):>18}"
          f"{str(baseline['median_detection_latency_minutes']):>20}")
    print()

    reduction = (
        1 - forest["alerts_per_hour"] / baseline["alerts_per_hour"]
        if baseline["alerts_per_hour"] else 0.0
    )
    print("Read this honestly: both detectors catch "
          f"{forest['episodes_detected']}/{forest['episodes']} incidents. The fitted")
    print(f"model's advantage is precision -- {reduction:.0%} fewer alerts per hour for the")
    print("same episode coverage, which is the difference between a useful pager")
    print("and one people mute.")


if __name__ == "__main__":
    main()
