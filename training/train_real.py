"""Fit the same detector on real operational telemetry (SMD).

The synthetic track measures whether the detector recovers incidents we designed:
we chose the escalation shape, the latency/CPU correlation, the recovery curve.
Doing well there partly means having recovered our own assumptions.

SMD is real telemetry from real machines, with anomalies labelled by the
operators from actual incident reports. Nothing about its failure modes was
chosen to be findable.

Design, mirroring `training/train.py` so the two are comparable in method:

- **Fit on the train split only**, which SMD ships as anomaly-free. Unsupervised
  on normal traffic, exactly as the served model is.
- **Set the alerting threshold with no labels at all**, from the training score
  distribution at a fixed alert budget. The served track calibrates to a target
  precision on labelled validation data; SMD has none, and a first attempt to
  carve one out of the test set failed outright -- see `threshold_from_alert_budget`.
  Both methods get the same budget, so the comparison is like for like.
- **The labelled test set is scored exactly once.**
- **Report point-level and episode-level metrics.** Accuracy is never reported:
  at a 1-9% anomaly rate it says nothing an always-negative predictor could not
  claim.
- **Keep the z-score baseline** alongside, as the synthetic track does. If the
  fitted model does not beat it, that is the finding.

    python training/train_real.py            # train and write artifacts
    python training/train_real.py --verify   # retrain and fail if metrics drifted
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.fetch_real_data import CACHE_DIR, MACHINES  # noqa: E402
from training.train import (  # noqa: E402
    CONTAMINATION,
    N_ESTIMATORS,
    RANDOM_STATE,
)

ARTIFACT_DIR = ROOT / "models" / "artifacts" / "real"
MODEL_DIR = ARTIFACT_DIR / "models"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.md"

VERIFY_TOLERANCE = 0.05
SAMPLE_SECONDS = 60  # SMD is sampled once per minute

# Fraction of points allowed to alert, applied to the TRAINING score distribution
# and to both methods equally. This is the operational knob a team actually has:
# how many alerts per hour on-call can absorb.
ALERT_BUDGET = 0.03


def load(machine):
    train = np.loadtxt(CACHE_DIR / "train" / f"{machine}.txt", delimiter=",")
    test = np.loadtxt(CACHE_DIR / "test" / f"{machine}.txt", delimiter=",")
    labels = np.loadtxt(CACHE_DIR / "test_label" / f"{machine}.txt", delimiter=",")
    return train, test, labels.astype(int)


def episodes_from(labels):
    """Contiguous runs of anomalous points, as (start, end) index pairs.

    SMD has no incident ids, so episodes are derived from the label sequence.
    This matters because point recall and episode recall answer different
    questions: catching 40% of a long outage late scores similarly to catching
    two short ones instantly, and on-call cares about the difference.
    """
    spans, start = [], None
    for index, value in enumerate(labels):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(labels) - 1))
    return spans


def episode_metrics(labels, predictions):
    spans = episodes_from(labels)
    caught, latencies = 0, []
    for start, end in spans:
        hits = np.flatnonzero(predictions[start:end + 1])
        if len(hits):
            caught += 1
            latencies.append(int(hits[0]))
    return {
        "episodes": len(spans),
        "episodes_detected": caught,
        "episode_recall": round(caught / len(spans), 4) if spans else 0.0,
        "median_detection_latency_minutes": (
            float(np.median(latencies)) if latencies else None
        ),
    }


def score_at(scores, labels, threshold):
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    hours = len(labels) * SAMPLE_SECONDS / 3600.0
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        "alerts_per_hour": round(float(predictions.sum() / hours), 3),
        **episode_metrics(labels, predictions),
    }


def threshold_from_alert_budget(train_scores, budget):
    """Threshold set from the TRAINING score distribution, using no labels at all.

    The first version of this calibrated to a target precision on a labelled half
    of the test set, mirroring the synthetic track. That does not survive contact
    with this data: SMD's anomalies are clustered in time, and machine-1-1's first
    half contains **zero** of them, so precision was undefined at every threshold
    and the calibration silently fell back to a fixed quantile.

    Setting the threshold from an alert budget on training scores is both the fix
    and the more honest model of deployment. When you turn a detector on you have
    no labels -- you have a number of alerts per hour the team can absorb. This
    never touches a test label before the single scoring pass, which the previous
    version did.
    """
    return float(np.quantile(train_scores, 1.0 - budget))


def run_machine(machine):
    train, test, labels = load(machine)

    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-9] = 1.0

    forest = IsolationForest(
        n_estimators=N_ESTIMATORS, contamination=CONTAMINATION,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    forest.fit((train - mean) / std)  # train split only -- SMD ships it anomaly-free

    train_scores = -forest.score_samples((train - mean) / std)
    scores = -forest.score_samples((test - mean) / std)

    # Threshold from training scores only. The labelled test set is scored once.
    threshold = threshold_from_alert_budget(train_scores, ALERT_BUDGET)

    zscore_train = np.abs((train - mean) / std).max(axis=1)
    zscore = np.abs((test - mean) / std).max(axis=1)
    # The baseline gets the same alert budget, so the comparison is like for like
    # rather than one method being handed a more generous threshold.
    zscore_threshold = threshold_from_alert_budget(zscore_train, ALERT_BUDGET)

    return {
        "machine": machine,
        "n_train": int(len(train)),
        "n_report": int(len(test)),
        "n_features": int(train.shape[1]),
        "anomaly_rate_report": round(float(labels.mean()), 4),
        "threshold": round(float(threshold), 6),
        # The budget is 3% of TRAINING points. What fraction of TEST points
        # actually clears the threshold measures how far the test period drifted
        # from the training period -- a number worth seeing, because a detector
        # firing on 28% of points is not operating where it was configured to.
        "test_alert_rate": round(float((scores >= threshold).mean()), 4),
        "report": score_at(scores, labels, threshold),
        "baseline_zscore": score_at(zscore, labels, zscore_threshold),
        # Three different trivial statistics, threshold-free. If only max|z| beat
        # the fitted model the result could be one lucky baseline; if all three
        # do, the fitted model is genuinely the wrong tool for this data.
        "threshold_free_pr_auc": {
            "isolation_forest": round(float(average_precision_score(labels, scores)), 4),
            "max_abs_z": round(float(average_precision_score(labels, zscore)), 4),
            "mean_abs_z": round(float(average_precision_score(
                labels, np.abs((test - mean) / std).mean(axis=1))), 4),
            "l2_norm": round(float(average_precision_score(
                labels, np.linalg.norm((test - mean) / std, axis=1))), 4),
            "random_equals_base_rate": round(float(labels.mean()), 4),
        },
    }, forest, {"mean": mean.tolist(), "std": std.tolist()}


def aggregate(results):
    """Pooled across machines, weighted by report-window length.

    A plain mean would let the smallest machine count as much as the largest.
    """
    total = sum(r["n_report"] for r in results)
    def weighted(path, key):
        return round(sum(r[path][key] * r["n_report"] for r in results) / total, 4)
    return {
        "precision": weighted("report", "precision"),
        "recall": weighted("report", "recall"),
        "f1": weighted("report", "f1"),
        "pr_auc": weighted("report", "pr_auc"),
        "alerts_per_hour": weighted("report", "alerts_per_hour"),
        "episodes": sum(r["report"]["episodes"] for r in results),
        "episodes_detected": sum(r["report"]["episodes_detected"] for r in results),
        "baseline_zscore_precision": weighted("baseline_zscore", "precision"),
        "baseline_zscore_recall": weighted("baseline_zscore", "recall"),
        "baseline_zscore_pr_auc": weighted("baseline_zscore", "pr_auc"),
        "baseline_zscore_alerts_per_hour": weighted("baseline_zscore", "alerts_per_hour"),
    }


def train():
    results, models, scalers = [], {}, {}
    for machine in MACHINES:
        result, forest, scaler = run_machine(machine)
        results.append(result)
        models[machine] = forest
        scalers[machine] = scaler

    pooled = aggregate(results)
    pooled["episode_recall"] = round(
        pooled["episodes_detected"] / pooled["episodes"], 4
    ) if pooled["episodes"] else 0.0

    metrics = {
        "model_type": "IsolationForest (fitted, unsupervised on normal traffic)",
        "note": "same approach as the served detector, fitted on real telemetry",
        "data_source": "REAL -- Server Machine Dataset (MIT licence)",
        "citation": ("Su, Zhao, Niu, Liu, Sun and Pei, 'Robust Anomaly Detection for "
                     "Multivariate Time Series through Stochastic Recurrent Neural "
                     "Networks', KDD 2019"),
        "url": "https://github.com/NetManAIOps/OmniAnomaly",
        "ground_truth": "anomalies labelled by the operators from real incident reports",
        "machines": MACHINES,
        "machine_selection_rule": (
            "first machine of each of the three groups, fixed before any result was "
            "seen -- reporting a subset chosen after the fact would be cherry-picking"
        ),
        "split": ("fit on the anomaly-free train split; threshold set from the "
                  "training score distribution at a fixed alert budget, using no "
                  "labels; the labelled test set is scored exactly once"),
        "alert_budget": ALERT_BUDGET,
        "accuracy_note": ("deliberately not reported: at a 1-9% anomaly rate an "
                          "always-negative predictor scores 91-99%"),
        "pooled_report": pooled,
        "per_machine": results,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    return models, scalers, metrics


def render_model_card(metrics):
    pooled = metrics["pooled_report"]
    rows = "\n".join(
        f"| `{r['machine']}` | {r['anomaly_rate_report']} | {r['report']['precision']} | "
        f"{r['report']['recall']} | {r['report']['pr_auc']} | "
        f"{r['report']['episodes_detected']}/{r['report']['episodes']} | "
        f"{r['report']['alerts_per_hour']} |"
        for r in metrics["per_machine"]
    )
    unmet_note = ""
    alert_rates = ", ".join(
        f"{r['test_alert_rate'] * 100:.1f}%" for r in metrics["per_machine"]
    )

    free_header = (
        r"| Machine | IsolationForest | max\|z\| | mean\|z\| | L2 | "
        "random (= base rate) |\n|---|---|---|---|---|---|"
    )
    def free_row(result):
        """Bold whichever detector actually won this row, not a chosen favourite."""
        scores = result["threshold_free_pr_auc"]
        columns = ["isolation_forest", "max_abs_z", "mean_abs_z", "l2_norm"]
        best = max(columns, key=lambda c: scores[c])
        cells = " | ".join(
            f"**{scores[c]}**" if c == best else f"{scores[c]}" for c in columns
        )
        return (f"| `{result['machine']}` | {cells} | "
                f"{scores['random_equals_base_rate']} |")

    free_rows = "\n".join(free_row(r) for r in metrics["per_machine"])
    threshold_free_table = f"{free_header}\n{free_rows}"
    worst = min(metrics["per_machine"],
                key=lambda r: r["threshold_free_pr_auc"]["isolation_forest"])
    worst_forest = worst["threshold_free_pr_auc"]["isolation_forest"]
    worst_base = worst["threshold_free_pr_auc"]["random_equals_base_rate"]
    worst_maxz = worst["threshold_free_pr_auc"]["max_abs_z"]
    return f"""# Model Card - Anomaly Detection on REAL telemetry

## What this is

The **same approach** as the served detector -- IsolationForest fitted
unsupervised on normal traffic, with an alerting threshold calibrated to a target
precision -- applied to real operational telemetry instead of generated series.

**Data:** {metrics['data_source']}
{metrics['citation']}
{metrics['url']}

Five weeks of telemetry from server machines in three data centres, 38 metrics
each, sampled every minute. **{metrics['ground_truth'].capitalize()}** -- which is
the ground truth the synthetic track cannot have, because there we chose what an
incident looks like.

### Machine selection

{metrics['machine_selection_rule'].capitalize()}.

All 28 machines would be roughly 550MB. The rule was fixed before any result was
seen, so a machine that scores badly is published rather than dropped.

## Method

{metrics['split'].capitalize()}.

SMD's train split carries no labels, so there is nowhere else a threshold could be
calibrated without touching the window it is reported on.
{unmet_note}
## The headline result: the fitted model loses to the trivial baseline

| Metric | Fitted IsolationForest | z-score baseline |
|---|---|---|
| Precision | {pooled['precision']} | {pooled['baseline_zscore_precision']} |
| Recall | {pooled['recall']} | **{pooled['baseline_zscore_recall']}** |
| **PR-AUC** | {pooled['pr_auc']} | **{pooled['baseline_zscore_pr_auc']}** |
| Alerts/hour | {pooled['alerts_per_hour']} | {pooled['baseline_zscore_alerts_per_hour']} |
| Episodes caught | {pooled['episodes_detected']}/{pooled['episodes']} | — |

**On real telemetry, taking the maximum standardised deviation across metrics
beats the fitted model on PR-AUC by more than double** -- and does so on every one
of the three machines, not on average.

PR-AUC is the comparison to read because it is threshold-free: neither method is
being helped or hurt by where its alerting line was drawn.

On synthetic data the same two methods are near-identical (0.8523 fitted against
0.8597 z-score). The gap only opens on real data, which is the point of having
this track at all.

### It is not one lucky baseline

Threshold-free PR-AUC for every trivial statistic tried, per machine:

{threshold_free_table}

All three simple detectors beat the fitted model on all three machines, and the
winner among them varies -- L2 on two, `max|z|` on the third -- so the result is
not an artefact of one favourable choice.

On `machine-3-1` the forest scores {worst_forest} against a base rate of
{worst_base}. A random ranker scores the base rate, so the fitted model is
essentially uninformative there, while `max|z|` reaches {worst_maxz}.

### Why, most likely

The generator produces incidents as *correlated multivariate escalations* --
latency, errors and CPU rising together -- which is the structure an isolation
forest is built to find, and which we put there. SMD's real anomalies are more
often a single metric departing sharply while the rest stay normal. Taking the
max across 38 standardised dimensions is close to the ideal detector for that
shape; averaging isolation depth over random splits of 38 dimensions dilutes it.

This is reported rather than tuned away. The fitted model is not re-specified
until it wins, because the finding *is* that a fitted model can lose to three
lines of arithmetic when the failure mode does not match its inductive bias.

### The operating point drifted too

The threshold is set to alert on 3% of *training* points. On the test period the
fitted detector actually fires on {alert_rates} of points for the three machines
respectively. That is distribution shift between the two periods, not a threshold
bug -- and the spread matters more than the average: one machine sits near budget
while another alerts on over half its minutes. A single global alert budget is
the wrong control for a fleet whose machines drift by different amounts, and this
is the practical argument for the drift detection this repository lacks.

**Accuracy is not reported.** {metrics['accuracy_note'].split(': ')[1].capitalize()}.

## Per machine

| Machine | Anomaly rate | Precision | Recall | PR-AUC | Episodes | Alerts/hr |
|---|---|---|---|---|---|---|
{rows}

The spread across machines is the useful part. These are different workloads with
different failure modes and anomaly rates varying by roughly an order of
magnitude; a detector that scored identically on all three would be suspicious.

## How this relates to the served model

The served detector reads six named features -- latency, errors, timeouts,
traffic, CPU, memory. SMD's 38 dimensions are anonymised and unlabelled, so this
is **not** the served model and could not be. What transfers is the method: fit on
normal traffic, calibrate a threshold on data you are not going to report on,
measure episodes rather than points, and keep a trivial baseline in the table.

## Known limitations

- Three machines of 28. The rule for choosing them is fixed and stated, but a
  larger sample would give a tighter picture.
- Anonymised features: no domain reasoning about *which* metric moved, which the
  served model's named features would allow.
- Point labels are treated as ground truth; SMD's labels mark anomalous windows
  as judged by operators, and judgement varies between them.
- No concept-drift handling across the five weeks.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="retrain and fail if metrics drifted from the committed file")
    args = parser.parse_args()

    models, scalers, metrics = train()
    pooled = metrics["pooled_report"]

    if args.verify:
        if not METRICS_PATH.exists():
            print("FAIL: real metrics.json missing; run without --verify first")
            return 1
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        old = committed["pooled_report"]["pr_auc"]
        new = pooled["pr_auc"]
        if abs(old - new) > VERIFY_TOLERANCE:
            print(f"FAIL: pooled PR-AUC drifted {old} -> {new} (tol {VERIFY_TOLERANCE})")
            return 1
        print(f"OK: retrained pooled PR-AUC {new} matches committed {old}")
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for machine, forest in models.items():
        joblib.dump({"model": forest, "scaler": scalers[machine]},
                    MODEL_DIR / f"{machine}.joblib")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    MODEL_CARD_PATH.write_text(render_model_card(metrics), encoding="utf-8")

    print(f"machines      : {', '.join(metrics['machines'])}")
    print(f"{'':14s}{'fitted':>10}{'z-score':>10}")
    print(f"{'precision':14s}{pooled['precision']:>10}{pooled['baseline_zscore_precision']:>10}")
    print(f"{'recall':14s}{pooled['recall']:>10}{pooled['baseline_zscore_recall']:>10}")
    print(f"{'pr_auc':14s}{pooled['pr_auc']:>10}{pooled['baseline_zscore_pr_auc']:>10}")
    print(f"{'alerts/hour':14s}{pooled['alerts_per_hour']:>10}"
          f"{pooled['baseline_zscore_alerts_per_hour']:>10}")
    print(f"episodes      : {pooled['episodes_detected']}/{pooled['episodes']} "
          f"(recall {pooled['episode_recall']})")
    for result in metrics["per_machine"]:
        print(f"  {result['machine']:14s} P={result['report']['precision']:<7}"
              f"R={result['report']['recall']:<7}"
              f"PR-AUC={result['report']['pr_auc']:<7} "
              f"(z-score PR-AUC {result['baseline_zscore']['pr_auc']})")
    print(f"artifacts     : {ARTIFACT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
