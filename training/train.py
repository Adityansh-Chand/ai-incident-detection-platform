"""Fit the anomaly detector and calibrate its alerting threshold.

Two things here differ from ordinary supervised training, because this is
time-series rare-event detection rather than tabular classification:

1. **The split is chronological, never random.** An incident spans consecutive
   minutes, so a random split puts minutes 5-8 of an episode in train and
   minutes 9-12 in test. The model then "detects" incidents it has already seen
   the middle of. Every reported number would be inflated. Data is split by
   time, per service: first 70% train, next 15% validation, last 15% test.

2. **The threshold is calibrated on validation, not guessed.** The detector
   produces a score; turning that into an alert requires a cut point, and the
   right cut point is an operations decision (how many false pages per hour is
   tolerable), not a modelling constant. It is chosen on the validation window
   to meet a target precision, and the test window is scored once with it.

`IsolationForest` is fitted on NORMAL training points only -- the labels are used
for calibration and evaluation, not for fitting, so this remains an unsupervised
detector. The z-score model the service previously shipped is retained as an
explicit baseline and reported alongside.

    python training/train.py             # fit and write models/artifacts/
    python training/train.py --verify    # refit and fail if metrics drifted
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.drift import build_reference  # noqa: E402

from pipeline.features import FEATURE_NAMES  # noqa: E402

DATA_PATH = ROOT / "datasets" / "telemetry.csv"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "anomaly_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
BASELINES_PATH = ARTIFACT_DIR / "service_baselines.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.md"
DRIFT_REFERENCE_PATH = ARTIFACT_DIR / "drift_reference.json"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# Operational target. This is a business decision, not a modelling constant:
# raising it means fewer false pages and more missed incidents. Exposed here so
# it can be changed deliberately and the effect re-measured.
TARGET_PRECISION = 0.75

N_ESTIMATORS = 300
CONTAMINATION = 0.03  # matches the generated anomaly rate
ZSCORE_THRESHOLD = 3.0
VERIFY_TOLERANCE = 0.05


def chronological_split(frame):
    """Split each service's series by time. No shuffling, ever."""
    train, validation, test = [], [], []
    for _, group in frame.groupby("service", sort=True):
        group = group.sort_values("observed_at")
        n = len(group)
        first = int(n * TRAIN_FRACTION)
        second = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
        train.append(group.iloc[:first])
        validation.append(group.iloc[first:second])
        test.append(group.iloc[second:])
    return (
        pd.concat(train).reset_index(drop=True),
        pd.concat(validation).reset_index(drop=True),
        pd.concat(test).reset_index(drop=True),
    )


def service_baselines(train):
    """Per-service mean/std from NORMAL training points only.

    Services have genuinely different healthy baselines -- payments idles at
    240ms, search at 95ms -- so one global threshold is wrong for all of them.
    Standardising per service is what lets a single model serve all four.
    """
    baselines = {}
    normal = train[train.label == 0]
    for service, group in normal.groupby("service", sort=True):
        stats = {}
        for name in FEATURE_NAMES:
            values = group[name].astype(float)
            std = float(values.std())
            stats[name] = {"mean": float(values.mean()), "std": std if std > 1e-9 else 1.0}
        baselines[service] = stats
    return baselines


def standardise(frame, baselines):
    matrix = np.zeros((len(frame), len(FEATURE_NAMES)), dtype=float)
    services = frame["service"].to_numpy()
    for column, name in enumerate(FEATURE_NAMES):
        values = frame[name].to_numpy(dtype=float)
        means = np.array([baselines[s][name]["mean"] for s in services])
        stds = np.array([baselines[s][name]["std"] for s in services])
        matrix[:, column] = (values - means) / stds
    return matrix


def episode_metrics(frame, predictions):
    """Was each incident caught at all, and how long did it take?

    Point-level recall understates a detector that catches every incident one
    minute late, and overstates one that catches half of a long episode and
    misses short ones entirely. On-call cares about episodes.
    """
    frame = frame.reset_index(drop=True).copy()
    frame["predicted"] = predictions

    caught, latencies = 0, []
    episodes = [e for e in frame.incident_id.unique() if e]
    for incident in episodes:
        block = frame[frame.incident_id == incident].sort_values("observed_at")
        hits = np.flatnonzero(block["predicted"].to_numpy())
        if len(hits):
            caught += 1
            latencies.append(int(hits[0]))  # minutes from episode start

    return {
        "episodes": len(episodes),
        "episodes_detected": caught,
        "episode_recall": round(caught / len(episodes), 4) if episodes else 0.0,
        "median_detection_latency_minutes": (
            float(np.median(latencies)) if latencies else None
        ),
    }


def alerts_per_hour(frame, predictions):
    minutes = len(frame)
    return round(float(predictions.sum()) / (minutes / 60.0), 3) if minutes else 0.0


def score_split(frame, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    labels = frame["label"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        "alerts_per_hour": alerts_per_hour(frame, predictions),
        **episode_metrics(frame, predictions),
    }


def calibrate(scores, labels, target_precision=TARGET_PRECISION):
    """Lowest threshold (so highest recall) that still meets target precision.

    Falls back to the best-F1 threshold when the target is unreachable, and says
    so, rather than silently shipping a threshold that misses the target.
    """
    candidates = np.unique(np.quantile(scores, np.linspace(0.50, 0.9995, 400)))
    best, fallback = None, None
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        if predictions.sum() == 0:
            continue
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        if fallback is None or f1 > fallback[1]:
            fallback = (float(threshold), float(f1))
        if precision >= target_precision:
            if best is None or recall > best[1]:
                best = (float(threshold), float(recall))

    if best is not None:
        return best[0], True
    return (fallback[0] if fallback else float(np.max(scores))), False


def train():
    frame = pd.read_csv(DATA_PATH).fillna({"incident_id": "", "severity": ""})
    train_set, validation_set, test_set = chronological_split(frame)

    baselines = service_baselines(train_set)
    x_train = standardise(train_set, baselines)
    x_validation = standardise(validation_set, baselines)
    x_test = standardise(test_set, baselines)

    # Unsupervised: fitted on normal training points only.
    normal_mask = train_set["label"].to_numpy() == 0
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train[normal_mask])

    # Higher score = more anomalous.
    # The score distribution on NORMAL training traffic is the drift reference:
    # what this detector expects a healthy period to look like.
    drift_reference = build_reference(-model.score_samples(x_train[normal_mask]))
    forest_validation = -model.score_samples(x_validation)
    forest_test = -model.score_samples(x_test)

    threshold, target_met = calibrate(
        forest_validation, validation_set["label"].to_numpy()
    )

    # Baseline: max absolute z-score, the model this service previously shipped
    # with hardcoded constants. Now at least fitted from train data.
    zscore_validation = np.abs(x_validation).max(axis=1)
    zscore_test = np.abs(x_test).max(axis=1)

    metrics = {
        "model_type": "IsolationForest (fitted, unsupervised on normal traffic)",
        "data_source": "synthetic -- training/generate_timeseries.py",
        "features": list(FEATURE_NAMES),
        "split": "chronological per service (no shuffling)",
        "n_train": len(train_set),
        "n_validation": len(validation_set),
        "n_test": len(test_set),
        "anomaly_rate_test": round(float(test_set["label"].mean()), 4),
        "target_precision": TARGET_PRECISION,
        "target_precision_met_on_validation": target_met,
        "threshold": round(float(threshold), 6),
        "contamination": CONTAMINATION,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "validation": score_split(validation_set, forest_validation, threshold),
        "test": score_split(test_set, forest_test, threshold),
        "baseline_zscore": {
            "threshold": ZSCORE_THRESHOLD,
            "test": score_split(test_set, zscore_test, ZSCORE_THRESHOLD),
        },
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    return model, baselines, metrics, drift_reference


def render_model_card(metrics):
    test = metrics["test"]
    base = metrics["baseline_zscore"]["test"]
    target_note = (
        f"met on validation" if metrics["target_precision_met_on_validation"]
        else "NOT reachable on validation; the best-F1 threshold was used instead"
    )
    return f"""# Model Card - Operational Anomaly Detection

## What this is

An `IsolationForest` fitted on **normal** training telemetry only, scoring each
minute of per-service metrics. Labels are used to calibrate the alerting threshold
and to evaluate -- not to fit -- so this remains an unsupervised detector.

## Training data - synthetic

Generated by `training/generate_timeseries.py` (seeded, reproducible). This is
**not** real production telemetry, and the model has never been validated against
real incidents. Incidents are multi-minute episodes with correlated degradation
across latency, errors, timeouts and CPU; benign traffic spikes are included as
hard negatives so that a detector keying on "unusual load" is penalised.

## Split

**Chronological, per service** -- first {int(TRAIN_FRACTION * 100)}% train,
next {int(VALIDATION_FRACTION * 100)}% validation, final
{100 - int((TRAIN_FRACTION + VALIDATION_FRACTION) * 100)}% test.

A random split would be invalid here. Incidents span consecutive minutes, so
shuffling places part of an episode in train and the rest in test, and the model
appears to detect incidents whose middle it has already seen.

## Measured performance (held-out test window, n={metrics['n_test']})

Anomalous points are {metrics['anomaly_rate_test']:.2%} of the test window.
**Accuracy is deliberately not reported**: always predicting "healthy" would score
about {1 - metrics['anomaly_rate_test']:.1%} and detect nothing.

| Metric | IsolationForest | z-score baseline |
|---|---|---|
| Precision | {test['precision']} | {base['precision']} |
| Recall (point) | {test['recall']} | {base['recall']} |
| F1 | {test['f1']} | {base['f1']} |
| PR-AUC | {test['pr_auc']} | {base['pr_auc']} |
| Episode recall | {test['episode_recall']} | {base['episode_recall']} |
| Episodes detected | {test['episodes_detected']}/{test['episodes']} | {base['episodes_detected']}/{base['episodes']} |
| Median detection latency | {test['median_detection_latency_minutes']} min | {base['median_detection_latency_minutes']} min |
| Alerts per hour | {test['alerts_per_hour']} | {base['alerts_per_hour']} |

**Episode recall is the number to read.** Point recall understates a detector that
catches every incident a minute late; on-call cares whether the incident was caught
at all, and how quickly.

## Threshold

Alerting threshold **{metrics['threshold']}**, calibrated on the validation window
for a target precision of {metrics['target_precision']} ({target_note}).

This is an operations dial, not a model parameter. Raising `TARGET_PRECISION` in
`training/train.py` yields fewer false pages and more missed incidents; lowering it
does the reverse. It is exposed rather than buried so the trade-off is deliberate.

## Known limitations

- Point-in-time scoring: each minute is judged independently, with no rolling
  window or trend features. A slow drift that never produces an unusual single
  minute will be missed.
- No seasonality model beyond what per-service standardisation absorbs.
- Per-service baselines are computed once at training time and never updated, so
  legitimate capacity changes would register as drift until refitting.
- Severity is generated and stored but not predicted.
- One seeded synthetic draw; no confidence intervals across seeds.
"""


def write_artifacts(model, baselines, metrics, drift_reference):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    BASELINES_PATH.write_text(
        json.dumps(
            {
                "note": "Per-service mean/std from NORMAL training points only.",
                "features": list(FEATURE_NAMES),
                "baselines": baselines,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    MODEL_CARD_PATH.write_text(render_model_card(metrics), encoding="utf-8")
    # The score distribution on normal training traffic, so the running detector
    # can tell whether it is still watching the same kind of system.
    DRIFT_REFERENCE_PATH.write_text(
        json.dumps(drift_reference, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    model, baselines, metrics, drift_reference = train()

    if args.verify:
        if not METRICS_PATH.exists():
            print("FAIL: metrics.json missing; run without --verify first")
            return 1
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        old, new = committed["test"]["f1"], metrics["test"]["f1"]
        if abs(old - new) > VERIFY_TOLERANCE:
            print(f"FAIL: test F1 drifted {old} -> {new} (tolerance {VERIFY_TOLERANCE})")
            return 1
        print(f"OK: retrained test F1 {new} matches committed {old}")
        return 0

    write_artifacts(model, baselines, metrics, drift_reference)
    test = metrics["test"]
    base = metrics["baseline_zscore"]["test"]
    print(f"split            : {metrics['split']}")
    print(f"train/val/test   : {metrics['n_train']}/{metrics['n_validation']}/{metrics['n_test']}")
    print(f"threshold        : {metrics['threshold']} (target precision "
          f"{metrics['target_precision']}, met={metrics['target_precision_met_on_validation']})")
    print()
    print(f"{'':18s}{'forest':>10s}{'z-score':>10s}")
    for key in ("precision", "recall", "f1", "pr_auc", "episode_recall", "alerts_per_hour"):
        print(f"{key:18s}{test[key]:>10}{base[key]:>10}")
    print(f"{'episodes':18s}{test['episodes_detected']:>7}/{test['episodes']:<2}"
          f"{base['episodes_detected']:>7}/{base['episodes']:<2}")
    print(f"{'detect latency':18s}{str(test['median_detection_latency_minutes']):>10}"
          f"{str(base['median_detection_latency_minutes']):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
