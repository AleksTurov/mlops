# Architecture and System Flow (EN)

### 1) Service roles
- **MLflow** — model registry and experiment tracking.
- **PostgreSQL (mlflow-db)** — MLflow metadata.
- **MinIO** — model artifacts (S3).
- **PostgreSQL (app-db)** — datasets and predictions.
- **Airflow** — schedules: data load, training, inference.
- **Model server** — online inference by MLflow alias.
- **Prometheus/Grafana** — metrics and dashboards.
- **Loki/Promtail** — log aggregation and search in Grafana.

### 2) Data flow
1. **Data load** → Airflow DAG `dag_data_predictions` stores data in **app-db**.
2. **Training** → DAG `dag_training` runs `ml.training.train_candidate()`, logs to **MLflow**, artifacts go to **MinIO**.
3. **Registry & alias** → DS manually sets alias in MLflow UI (e.g., `Production`).
4. **Serving** → `model-server` reads alias and loads model from MLflow/MinIO.
5. **Inference** → DAG `dag_inference` runs `ml.inference.run_inference()` and writes predictions to **app-db**.
6. **Model monitoring** → DAG `dag_model_monitoring` runs `ml.training.evaluate_models()`.
7. **Observability** → `model-server` exposes `/metrics`, Prometheus/Grafana visualize.
8. **Logs** → Promtail ships Docker logs to Loki, Grafana shows logs.

### 3) Mini demo for an article (working path)
1. Bring up services with `docker compose`.
2. Run DAG `dag_data_predictions` (loads iris into app-db).
3. Run DAG `dag_training` (trains multiple baseline models and logs metrics in MLflow).
4. Set alias `Production` in MLflow UI.
5. Send a request via `scripts/predict_request.py`.
6. Show `Tag watcher` and Grafana for status/metrics.

All ports are configured via `.env` (`*_PORT`).

Dashboards are provisioned from `monitoring/grafana/dashboards-min`.

### 4) Databases
By default the stack uses **separate Postgres instances** (`mlflow-db`, `airflow-db`, `app-db`).
This keeps isolation and avoids accidental cross‑service schema conflicts.

**Single Postgres is possible** (not enabled by default):
- Use one Postgres and separate schemas/databases for MLflow, Airflow, and app data.
- Configure `MLFLOW_POSTGRES_URI`, `AIRFLOW_DB_URI`, `APP_DB_URI` accordingly.
- Docker Compose already uses a shared network, so services talk to each other over internal DNS.
