"""Quality gate for the real-data track (SMD).

This gate is unusual: the headline result is that the **fitted model loses to a
trivial baseline**. So the tests protect the honesty of that comparison rather
than a performance floor. If someone later tunes the forest until it wins, these
should be updated deliberately -- not silently satisfied.

Skips cleanly when the dataset is not cached, so a fresh clone stays green.

Observed at the time of writing (pooled, threshold-free PR-AUC):
    IsolationForest 0.1897   z-score 0.4348
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.fetch_real_data import CACHE_DIR, MACHINES  # noqa: E402

METRICS_PATH = ROOT / "models" / "artifacts" / "real" / "metrics.json"

pytestmark = pytest.mark.skipif(
    not (CACHE_DIR / "train" / f"{MACHINES[0]}.txt").exists()
    or not METRICS_PATH.exists(),
    reason="SMD not cached; run training/fetch_real_data.py then train_real.py",
)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_machine_selection_rule_is_recorded(metrics):
    """The rule was fixed before results were seen; the artifact must say so."""
    assert metrics["machines"] == MACHINES
    assert "cherry-picking" in metrics["machine_selection_rule"]


def test_threshold_uses_no_labels(metrics):
    """The whole method rests on this: no test label is touched before scoring."""
    assert "no labels" in metrics["split"]
    assert metrics["alert_budget"] > 0


def test_accuracy_is_never_reported(metrics):
    """At a 1-9% anomaly rate, accuracy is a misleading summary."""
    assert "accuracy_note" in metrics
    for result in metrics["per_machine"]:
        assert "accuracy" not in result["report"]
        assert "accuracy" not in result["baseline_zscore"]


def test_every_machine_is_published(metrics):
    """No machine may be dropped for scoring badly."""
    assert len(metrics["per_machine"]) == len(MACHINES)
    reported = {r["machine"] for r in metrics["per_machine"]}
    assert reported == set(MACHINES)


def test_baseline_comparison_is_present_and_fair(metrics):
    """Both methods must be scored at the same alert budget."""
    for result in metrics["per_machine"]:
        assert "baseline_zscore" in result
        free = result["threshold_free_pr_auc"]
        for key in ("isolation_forest", "max_abs_z", "mean_abs_z", "l2_norm"):
            assert key in free


def test_the_negative_finding_is_still_the_finding(metrics):
    """The fitted model loses. If that stops being true, this must be revisited.

    Deliberately asserted rather than left implicit: the value of this track is
    the honest negative result, and a silent reversal -- from tuning, or from a
    library change -- should surface as a failing test and a conscious rewrite of
    the model card, not as a quietly better number.
    """
    pooled = metrics["pooled_report"]
    assert pooled["pr_auc"] < pooled["baseline_zscore_pr_auc"], (
        "the fitted detector now beats the z-score baseline on real telemetry. "
        "That may be genuine progress, but the model card states the opposite -- "
        "update it deliberately rather than letting the claim rot."
    )


def test_forest_loses_on_every_machine_not_just_on_average(metrics):
    """Pooled results can hide a split decision; this asserts it is not one."""
    for result in metrics["per_machine"]:
        free = result["threshold_free_pr_auc"]
        best_simple = max(free["max_abs_z"], free["mean_abs_z"], free["l2_norm"])
        assert free["isolation_forest"] < best_simple, (
            f"{result['machine']}: fitted model now beats every simple statistic"
        )


def test_real_data_is_attributed(metrics):
    assert "REAL" in metrics["data_source"]
    assert metrics["citation"]
    assert "OmniAnomaly" in metrics["url"]
    assert "incident reports" in metrics["ground_truth"]
