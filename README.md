# MLOps Reference Stack (MLflow + Airflow + Prometheus)

This repository provides a minimal, script-first MLOps stack for data scientists (Phase 1):
- you run experiments locally or in notebooks,
- every training run is logged to MLflow (params/metrics/artifacts),
- artifacts are stored in MinIO,
- a single model service auto-follows MLflow alias from env,
- Prometheus/Grafana monitor all services.

Key components
- MLflow + PostgreSQL for tracking and model registry
- MinIO for artifact storage (S3-compatible)
- Airflow for batch and scheduled tasks only
- Model server (reads MLflow alias from env)
- Tag watcher (reads MLflow aliases and reports status)
- Prometheus + Grafana

Docs
- Demo guide (EN): [docs/DEMO.md](docs/DEMO.md)
- Architecture and end-to-end flow (EN): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Scripts & DAGs (EN): [docs/SCRIPTS.md](docs/SCRIPTS.md)

Quick start
```bash
cp env.dev.example .env
docker compose --env-file .env up -d --build
```

Demo flow (for the article)
1) Bring up services via Docker Compose.
2) Run Airflow DAG `dag_data_predictions` (loads an open dataset into app-db).
3) Run Airflow DAG `dag_training` (trains multiple models, logs metrics in MLflow).
4) In MLflow UI, set alias `Production` for the best model.
5) `model-server` auto-loads the `Production` alias.
6) Prometheus/Grafana start tracking metrics from the beginning.
7) Run Airflow DAG `dag_inference` to send predictions using the `Production` alias.
8) Observe request/latency metrics in Grafana.

Main endpoints (ports are defined in .env)
- MLflow UI: http://localhost:${MLFLOW_PORT}
- Airflow UI: http://localhost:${AIRFLOW_WEB_PORT}
- MinIO Console: http://localhost:${MINIO_CONSOLE_PORT}
- Model service: http://localhost:${MODEL_SERVER_PORT} (alias via `MODEL_ALIAS`)
- Tag watcher: http://localhost:${TAG_WATCHER_PORT}/status
- Grafana: http://localhost:${GRAFANA_PORT}
- Prometheus: http://localhost:${PROMETHEUS_PORT}

Workflow (data scientist view)
1) Build features in notebooks, Airflow, or service.
2) Train locally and log to MLflow (params/metrics/artifacts).
3) Assign alias (for example, `Production`) to the candidate model.
4) `model-server` auto-loads the alias defined in `MODEL_ALIAS`.
5) When ready, promote by switching MLflow alias to a new version.

Why MLflow here?
MLflow provides model registry, run metadata, metrics comparison, and artifact storage. Even if you upload datasets directly, MLflow gives reproducibility, auditability, and easy promotion/rollback via aliases.

Python toolkit
Install and use the CLI from [README_library.md](README_library.md) to automate MLflow aliases (Phase 1).

Predict request (demo):
```bash
python scripts/predict_request.py --url http://localhost:${MODEL_SERVER_PORT} --payload data/predict_payload.json
```

Notes
- Airflow is kept for scheduled/batch workflows only (daily/weekly retraining or batch predictions).
- Online inference is performed via the single model service.

