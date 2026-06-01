# AI Incident Detection Platform

Operational anomaly detection service for scoring telemetry events using
domain-specific features and a lightweight z-score anomaly model.

## Pipeline

```mermaid
flowchart LR
  Telemetry --> FeatureExtraction
  FeatureExtraction --> AnomalyModel
  AnomalyModel --> Score
  Score --> Evaluation
```

## API

- `GET /health`
- `GET /metrics`
- `GET /events` protected when `API_KEY` is set
- `POST /score`

See `DEMO.md` for terminal demo steps, curl commands, and sample request/response files.

Example:

```json
{
  "service": "checkout",
  "latency_ms": 900,
  "error_count": 20,
  "timeout_count": 6,
  "traffic_rpm": 2300
}
```

Set `API_KEY` to require `X-API-Key` on scoring/event endpoints.
Set `APP_DB_PATH` to control the SQLite event database location.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
python evaluation/evaluate.py
uvicorn api.server:app --reload --port 8000
```

With the server running, use a second terminal for the smoke check:

```bash
python scripts/smoke_test.py
```

Docker:

```bash
cp .env.example .env
docker compose up --build
```

Kubernetes manifests live in `k8s/deployment.yaml` and include probes, resource
limits, a Service, and a PVC for the SQLite event store. The default manifest
uses one replica because SQLite is the default event store.

Dockerfile, Docker Compose, and Kubernetes configuration are validated by static
inspection/YAML parsing in this workspace. Runtime container and cluster
validation remains a CI or cloud-environment step.

## Reviewer Status

- Purpose: operational telemetry scoring for anomaly and incident signals.
- Quickstart: run tests/eval, start `uvicorn api.server:app --reload --port 8000`, then run `python scripts/smoke_test.py`.
- Demo path: use `DEMO.md` for curl examples and sample request/response files.
- Deployment status: local tests and smoke tests pass; Docker/Compose/Kubernetes config is present; Docker image builds are validated in CI; cloud deployment is pending.
- Remaining gaps: production telemetry ingestion, alert routing, managed auth/secrets, observability, cloud deployment, and production data governance.
- Portfolio index: https://github.com/Adityansh-Chand/ai-engineering-portfolio

## Highlights

- Telemetry features for latency, errors, timeouts, and traffic.
- Synthetic labeled time-series generator.
- Fitted baseline anomaly model.
- Evaluation script over bundled labeled sample data.
- SQLite event audit trail for score results.
- GitHub Actions CI for tests, eval, and container build.
- Production data contract in `datasets/production_schema.json`.

## License

MIT
