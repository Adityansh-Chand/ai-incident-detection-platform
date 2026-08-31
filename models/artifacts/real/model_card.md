# Model Card - Anomaly Detection on REAL telemetry

## What this is

The **same approach** as the served detector -- IsolationForest fitted
unsupervised on normal traffic, with an alerting threshold calibrated to a target
precision -- applied to real operational telemetry instead of generated series.

**Data:** REAL -- Server Machine Dataset (MIT licence)
Su, Zhao, Niu, Liu, Sun and Pei, 'Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Networks', KDD 2019
https://github.com/NetManAIOps/OmniAnomaly

Five weeks of telemetry from server machines in three data centres, 38 metrics
each, sampled every minute. **Anomalies labelled by the operators from real incident reports** -- which is
the ground truth the synthetic track cannot have, because there we chose what an
incident looks like.

### Machine selection

First machine of each of the three groups, fixed before any result was seen -- reporting a subset chosen after the fact would be cherry-picking.

All 28 machines would be roughly 550MB. The rule was fixed before any result was
seen, so a machine that scores badly is published rather than dropped.

## Method

Fit on the anomaly-free train split; threshold set from the training score distribution at a fixed alert budget, using no labels; the labelled test set is scored exactly once.

SMD's train split carries no labels, so there is nowhere else a threshold could be
calibrated without touching the window it is reported on.

## The headline result: the fitted model loses to the trivial baseline

| Metric | Fitted IsolationForest | z-score baseline |
|---|---|---|
| Precision | 0.1481 | 0.134 |
| Recall | 0.4547 | **0.7196** |
| **PR-AUC** | 0.1897 | **0.4348** |
| Alerts/hour | 16.6572 | 18.2959 |
| Episodes caught | 21/25 | — |

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

| Machine | IsolationForest | max\|z\| | mean\|z\| | L2 | random (= base rate) |
|---|---|---|---|---|---|
| `machine-1-1` | 0.3796 | 0.4451 | 0.4149 | **0.4766** | 0.0946 |
| `machine-2-1` | 0.159 | 0.1857 | 0.1789 | **0.2077** | 0.0494 |
| `machine-3-1` | 0.0266 | **0.6301** | 0.3851 | 0.5589 | 0.0107 |

All three simple detectors beat the fitted model on all three machines, and the
winner among them varies -- L2 on two, `max|z|` on the third -- so the result is
not an artefact of one favourable choice.

On `machine-3-1` the forest scores 0.0266 against a base rate of
0.0107. A random ranker scores the base rate, so the fitted model is
essentially uninformative there, while `max|z|` reaches 0.6301.

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
fitted detector actually fires on 21.1%, 6.2%, 52.2% of points for the three machines
respectively. That is distribution shift between the two periods, not a threshold
bug -- and the spread matters more than the average: one machine sits near budget
while another alerts on over half its minutes. A single global alert budget is
the wrong control for a fleet whose machines drift by different amounts, and this
is the practical argument for the drift detection this repository lacks.

**Accuracy is not reported.** At a 1-9% anomaly rate an always-negative predictor scores 91-99%.

## Per machine

| Machine | Anomaly rate | Precision | Recall | PR-AUC | Episodes | Alerts/hr |
|---|---|---|---|---|---|---|
| `machine-1-1` | 0.0946 | 0.2421 | 0.5397 | 0.3796 | 5/8 | 12.651 |
| `machine-2-1` | 0.0494 | 0.2011 | 0.2504 | 0.159 | 12/13 | 3.69 |
| `machine-3-1` | 0.0107 | 0.0111 | 0.539 | 0.0266 | 4/4 | 31.338 |

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
