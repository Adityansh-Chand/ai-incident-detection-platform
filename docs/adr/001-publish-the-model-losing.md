# ADR-001 — Publish the fitted model losing to arithmetic

**Status:** Accepted · **Date:** 2026-06

## Context

On synthetic telemetry the fitted `IsolationForest` and a three-line z-score baseline are
near-identical (PR-AUC 0.8523 against 0.8597). On the real
[Server Machine Dataset](https://github.com/NetManAIOps/OmniAnomaly) — five weeks of
telemetry from three data centres, anomalies labelled by operators from actual incident
reports — the gap opens the wrong way:

| Metric | Fitted IsolationForest | z-score baseline |
|---|---|---|
| **PR-AUC** | 0.1897 | **0.4348** |
| Recall | 0.4547 | **0.7196** |

Per machine, against every trivial statistic tried:

| Machine | IsolationForest | max\|z\| | mean\|z\| | L2 | random (= base rate) |
|---|---|---|---|---|---|
| `machine-1-1` | 0.3796 | 0.4451 | 0.4149 | **0.4766** | 0.0946 |
| `machine-2-1` | 0.1590 | 0.1857 | 0.1789 | **0.2077** | 0.0494 |
| `machine-3-1` | 0.0266 | **0.6301** | 0.3851 | 0.5589 | 0.0107 |

Every simple statistic beats the fitted model on every machine, and the winner among them
varies — so this is not one cherry-picked baseline. On `machine-3-1` the forest scores
0.0266 against a base rate of 0.0107, which is essentially uninformative.

The machines were fixed in advance by a stated rule (the first of each of the three groups),
so this is not a bad draw that could have been reshuffled.

## Decision

Publish it, as the headline of the real-data section, with the per-machine table.

Keep the fitted model as the served detector, on the narrower claim the synthetic track
actually supports: **32% fewer alerts per hour for identical incident coverage** (2.639
against 3.869, both catching 17/17 episodes within a minute).

## Alternatives considered

**Report only the synthetic track, where the model does fine.** The result would be a
service whose README says a fitted detector works, with nothing contradicting it. Rejected:
the real-data track exists precisely to catch the case where recovering our own generator
was mistaken for capability, and deleting it the first time it does its job would make every
other number in the portfolio worth less.

**Tune the forest until it wins** — contamination, estimators, feature engineering, per-machine
models. Rejected on principle and on diagnosis. The cause is visible in what the two data
sources are: the generator produces incidents as *correlated multivariate escalations*, which
is the structure an isolation forest is built to find, and real server anomalies frequently
are not that. Tuning until the number improves would be fitting the test set through a human,
and the honest fix is a different method rather than a better-tuned wrong one.

**Switch the served model to the z-score baseline.** The most defensible alternative, and
the closest call in this record. Rejected because the synthetic track — which is the one
matching this service's actual input schema — shows the fitted model buying a 32% alert
reduction at equal coverage, and alert volume is the difference between a pager people act
on and one they mute. The real-data track uses 38 metrics per machine; the service scores
six. They are not the same problem, and the losing result is evidence about the *method*,
not a benchmark of the deployed model.

**Report it in a limitations footnote.** Rejected. It is the most informative result in the
repository, and burying the most informative result to protect a claim is the failure mode
this whole portfolio was rebuilt to remove.

## Consequences

- The README leads its real-data section with the service's own model losing. That reads
  badly at a glance and is the point at depth.
- The quality gate asserts precision and alert volume and **deliberately does not assert a
  PR-AUC win** — a test asserting something the data does not support would have to be
  weakened later, and a weakened test is worse than an absent one.
- Accuracy is reported nowhere. At a 4.2% base rate, predicting "healthy" every minute
  scores 95.8% and detects nothing.
- The finding is bounded honestly: 38 metrics against six, three machines, one dataset. It
  says isolation forests were not the right tool *here*, not that fitted anomaly detection
  is a bad idea.

## Revisit when

A method is tried that suits the real data's structure — a temporal model, or per-machine
normalisation — and can be compared on the same held-out split. The harness to do that is
already in place, and the baseline to beat is now on record rather than assumed.
