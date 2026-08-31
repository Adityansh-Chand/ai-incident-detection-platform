# ADR-002 — Set the threshold from an alert budget, not a precision target

**Status:** Accepted · **Date:** 2026-06

## Scope

This governs the **real-data track**. The served synthetic track keeps target-precision
calibration (`TARGET_PRECISION = 0.75` in `training/train.py`), because it has a labelled
validation window with positives in it and the alert-budget argument below does not apply.
The two are stated separately because they genuinely differ, and describing both as
"calibrated" without saying how would hide the more interesting half.

## Context

An anomaly score needs a threshold. The natural approach, and the one the served track uses,
calibrates it on a labelled validation window to hit a target operational precision — pick
the score at which precision reaches 0.75, and alert above it.

That approach failed silently on real data. `machine-1-1`'s calibration half contained
**zero labelled anomalies**. Precision is undefined with no positives, so the calibration
had nothing to optimise and quietly fell back to a default. The threshold looked calibrated,
was reported as calibrated, and was not.

The failure is not specific to that machine. It is inherent to the method: **calibrating on
precision requires labels in the calibration window**, and the defining property of rare
events is that a given window may contain none.

## Decision

On the real-data track, set the threshold from a fixed **alert budget** on the *training*
score distribution, using no labels at all: allow 3% of training points to alert, and take
the score at that quantile. The labelled test set is then scored exactly once.

Give the z-score baseline the **same budget**, so the comparison in ADR-001 is like for like
rather than two methods at two operating points.

## Alternatives considered

**Target precision, with a guard for the zero-positive case.** The minimal fix — detect the
empty case and fall back explicitly rather than silently. Rejected because the fallback
would still have to be *something*, and whatever that something is becomes the real
threshold on exactly the windows where calibration was needed most. Guarding the symptom
leaves the method depending on a condition it cannot guarantee.

**Target recall instead.** Same defect for the same reason: no positives, no recall.

**A fixed score threshold.** No labels needed, and no meaning either — an isolation forest's
score scale depends on the fitted data, so one constant cannot transfer across machines or
across retraining runs.

**Youden's J or F1-maximising threshold on a labelled validation set.** Statistically better
when labels exist. Rejected on the same availability argument, plus a deployment one:
production has an alert budget (how many pages an on-call rotation tolerates) far more
reliably than it has labelled anomalies.

## Consequences

- The threshold is set by a number an operations team actually has an opinion about — how
  many alerts per hour is acceptable — rather than one requiring labelled incidents nobody
  has yet.
- **The budget does not hold under distribution shift, and this is now measurable rather
  than hidden.** The threshold allows 3% of training points to alert; on the test period the
  detector fires on **21.1%, 6.2% and 52.2%** of points across the three machines. That is
  the method being honest about drift: one machine is roughly at budget while another alerts
  on over half its minutes.
- The spread matters more than the average, and it is the concrete argument for per-machine
  or drift-triggered recalibration — a gap this repository names rather than closes.
- Calibration and evaluation are cleanly separated: the budget comes from training scores,
  the labels are touched once. Nothing can leak backwards.
- The same reasoning was reused in the RAG service for
  [abstention thresholds](https://github.com/Adityansh-Chand/enterprise-rag-knowledge-system/blob/main/docs/adr/005-abstention-signal.md),
  fitted on answerable queries only. Production rarely supplies labelled negatives; it
  always supplies the positive distribution.

## Revisit when

Labelled incidents accumulate in production at a rate that makes a labelled calibration
window reliably non-empty. At that point precision targeting becomes available again — and
the alert budget remains the right *constraint* even then, because it is the one the pager
rotation is actually bounded by.
