# Architecture and System Flow (EN)

## 1) Service roles
- **MLflow** — model registry and experiment tracking.
- **PostgreSQL (mlflow-db)** — MLflow metadata.
- **MinIO** — model artifacts (S3-compatible).
- **PostgreSQL (app-db)** — datasets and predictions.
- **Airflow** — batch pipelines: data load, training, inference, monitoring.
- **MLflow Autoserve** — watcher that starts MLflow Serve for each alias.
- **MLflow Serve containers** — online inference per model+alias.
- **Prometheus/Grafana** — health/metrics dashboards.
- **Loki/Promtail** — log aggregation and search in Grafana.
- **Blackbox exporter** — HTTP health probes.

## 2) Data flow (end-to-end)
1. **Data load** → Airflow DAG `dag_data_predictions` stores data in **app-db**.
2. **Training** → DAG `dag_training` runs `ml.training.train_candidate()`, logs runs to **MLflow**, artifacts go to **MinIO**.
3. **Registry & alias** → training auto-assigns alias `Production` to the best version (can be adjusted in MLflow UI).
4. **Serving** → `mlflow-autoserve` detects aliases and starts `mlflow models serve` containers.
5. **Inference (batch)** → DAG `dag_inference` loads the latest dataset and writes predictions to **app-db**.
6. **Model monitoring** → DAG `dag_model_monitoring` compares candidate vs production.
7. **Observability** → Blackbox checks `/ping` for each `mlflow-serve-*` container; Grafana shows health status.
8. **Logs** → Promtail ships Docker logs to Loki; Grafana shows logs by service/container.

## 3) Serving flow (MLflow Serve)
- Each alias spawns a dedicated container named like `mlflow-serve-<model>-<alias>`.
- Health endpoint: `GET /ping` (inside Docker network).
- Inference endpoint: `POST /invocations` with MLflow scoring format.
- Expected input schema is stored in MLflow artifacts: `data_contract/input_schema.json` and `sample_input.csv`.
- MLflow Serve does **not** expose Prometheus `/metrics`; use Blackbox health probes for availability.

## 4) Observability
- **Service health**: `probe_success` from Blackbox.
- **Dashboards**: provisioned from `monitoring/grafana/dashboards-min`.
	- **Service Health Detailed** (all services + model/alias status)
	- **MLflow Serving** (model alias health and probe latency)
- **Logs**: Loki via Promtail; filter by `container` or `service` labels.

## 5) Databases and network
By default the stack uses **separate Postgres instances** (`mlflow-db`, `airflow-db`, `app-db`).
This keeps isolation and avoids accidental cross-service schema conflicts.

**Single Postgres is possible** (not enabled by default):
- Use one Postgres and separate schemas/databases for MLflow, Airflow, and app data.
- Configure `MLFLOW_POSTGRES_URI`, `AIRFLOW_DB_URI`, `APP_DB_URI` accordingly.
- Docker Compose already uses a shared network, so services talk to each other over internal DNS.

## 6) Endpoints
All ports are configured via `.env` (`*_PORT`).
- MLflow UI: http://localhost:${MLFLOW_PORT}
- Airflow UI: http://localhost:${AIRFLOW_WEB_PORT}
- MinIO Console: http://localhost:${MINIO_CONSOLE_PORT}
- Grafana: http://localhost:${GRAFANA_PORT}
- Prometheus: http://localhost:${PROMETHEUS_PORT}
- Loki: http://localhost:${LOKI_PORT}
