
# AI Incident Detection Platform

ML pipeline for anomaly detection using synthetic time-series simulation.

## Architecture

```mermaid
flowchart LR
Logs --> FeatureExtraction
FeatureExtraction --> Model
Model --> AnomalyScore
AnomalyScore --> Evaluation
```

## Pipeline
data → features → anomaly score → evaluate

### Highlights
synthetic dataset generator
evaluation-aware pipeline
modular ML structure

## License
MIT
