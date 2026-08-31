# ADR-003 — Score episodes and alert volume, never accuracy

**Status:** Accepted · **Date:** 2026-04

## Context

Anomalies here are **incident episodes** — consecutive minutes of correlated latency, error
and CPU escalation followed by recovery — not isolated points. The held-out period contains
17 incidents across 6,048 minutes, 4.20% of points anomalous.

At that base rate, a detector that predicts "healthy" every single minute scores **95.8%
accuracy** and detects nothing. Accuracy is not merely uninformative here; it is
anti-informative, because the degenerate model wins it.

Point-level precision and recall are better but still mismatched to the question. An
operator does not care whether all seven minutes of an incident were flagged. They care
whether they were paged, how quickly, and how many times they were paged for nothing.

## Decision

Report, in this order: **episode recall** (was the incident caught anywhere in its window),
**median detection latency**, **alerts per hour**, then point-level precision/recall/F1 and
PR-AUC.

**Do not report accuracy anywhere**, and say in the README why.

## Alternatives considered

**Report accuracy alongside everything else, for completeness.** The instinct that more
numbers is more honest. Rejected: a 95.8% figure sitting in a table will be read as the
headline no matter what surrounds it, and it describes a model that does nothing. Omitting
it and explaining the omission conveys more than including it and explaining the caveat.

**Point-level F1 as the headline.** Conventional for anomaly detection benchmarks and what
most papers report. Rejected because it makes a detector that fires on every minute of a
long incident look better than one that fires once at the start, and the second is the more
useful pager. It optimises for the wrong thing at the operating point.

**Episode-level only.** Cleanest match to the operational question. Rejected because it
hides alert volume entirely — a detector that alerts constantly catches every episode by
definition, and episode recall alone cannot distinguish it from a good one. **Episode recall
and alerts-per-hour have to be read together or neither means anything.**

**Time-to-detect as the single metric.** Rejected for the same reason in reverse: it says
nothing about the incidents never caught at all.

## Consequences

- The headline claim is narrow and survivable: 32% fewer alerts per hour (2.639 against
  3.869) for identical episode coverage (17/17 both, median 1 minute both). Not "ML beats
  the baseline" — the baseline actually edges the fitted model on PR-AUC (0.8597 against
  0.8523), and that is reported in the same table.
- "Caught anywhere in the window" is a definition with judgement in it, so it lives in
  config rather than as a buried constant, and the model card records the reasoning.
- The quality gate asserts precision and alert volume specifically. It does **not** assert a
  PR-AUC win, because the data does not support one.
- Comparison with published anomaly-detection results is harder, since most report
  point-level F1. Accepted: matching a benchmark convention that rewards the wrong behaviour
  is not a reason to adopt it.

## Revisit when

The service acquires a real alert consumer with a real escalation policy. Then
alerts-per-hour becomes a budget with a number attached rather than a comparative figure,
and detection latency becomes an SLO rather than a measurement.
