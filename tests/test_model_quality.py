"""Quality gate: fails if the detector regresses.

Metrics are recomputed from the artifact against the held-out window rather than
read from metrics.json, so a stale metrics file cannot make a broken detector
look healthy.

Observed at the time of writing: precision 0.7895, recall 0.8268, F1 0.8077,
PR-AUC 0.8523, episode recall 1.0 (17/17), 2.639 alerts/hour.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.anomaly_model import load_artifacts  # noqa: E402
from training.train import (  # noqa: E402
    ZSCORE_THRESHOLD,
    chronological_split,
    score_split,
    standardise,
)

MIN_PRECISION = 0.65
MIN_F1 = 0.70
MIN_PR_AUC = 0.75
MIN_EPISODE_RECALL = 0.85
# An alert every few minutes is a pager people mute. This is an operational
# ceiling, not a statistical one.
MAX_ALERTS_PER_HOUR = 6.0


@pytest.fixture(scope="module")
def evaluated():
    model, baselines, metadata = load_artifacts()
    frame = pd.read_csv(ROOT / "datasets" / "telemetry.csv").fillna(
        {"incident_id": "", "severity": ""}
    )
    _, _, test = chronological_split(frame)
    x_test = standardise(test, baselines)
    forest = score_split(test, -model.score_samples(x_test), float(metadata["threshold"]))
    baseline = score_split(test, np.abs(x_test).max(axis=1), ZSCORE_THRESHOLD)
    return metadata, forest, baseline, test


def test_precision_meets_bar(evaluated):
    _, forest, _, _ = evaluated
    assert forest["precision"] >= MIN_PRECISION


def test_f1_meets_bar(evaluated):
    _, forest, _, _ = evaluated
    assert forest["f1"] >= MIN_F1


def test_pr_auc_meets_bar(evaluated):
    _, forest, _, _ = evaluated
    assert forest["pr_auc"] >= MIN_PR_AUC


def test_most_incidents_are_detected(evaluated):
    _, forest, _, _ = evaluated
    assert forest["episode_recall"] >= MIN_EPISODE_RECALL


def test_alert_volume_stays_operationally_sane(evaluated):
    _, forest, _, _ = evaluated
    assert forest["alerts_per_hour"] <= MAX_ALERTS_PER_HOUR


def test_fitted_model_beats_the_zscore_baseline_on_precision(evaluated):
    """The specific claim the README makes -- asserted, not just stated.

    Note it is deliberately NOT a claim about PR-AUC: the baseline is
    competitive there, and the fitted model's advantage is at the operating
    point, not in raw ranking.
    """
    _, forest, baseline, _ = evaluated
    assert forest["precision"] > baseline["precision"]
    assert forest["alerts_per_hour"] < baseline["alerts_per_hour"]


def test_test_window_is_rare_event(evaluated):
    """If anomalies stopped being rare, the metric choices no longer make sense."""
    _, _, _, test = evaluated
    assert 0.01 < test["label"].mean() < 0.10


def test_evaluation_window_holds_enough_incidents(evaluated):
    """Episode recall over 2-3 incidents would not mean anything."""
    _, forest, _, _ = evaluated
    assert forest["episodes"] >= 10


def test_committed_metrics_match_the_artifact(evaluated):
    metadata, forest, _, _ = evaluated
    assert forest["f1"] == pytest.approx(metadata["test"]["f1"], abs=0.01)


def test_split_is_declared_chronological(evaluated):
    metadata, _, _, _ = evaluated
    assert "chronological" in metadata["split"]


def test_data_provenance_is_declared_as_synthetic(evaluated):
    metadata, _, _, _ = evaluated
    assert "synthetic" in metadata["data_source"].lower()
