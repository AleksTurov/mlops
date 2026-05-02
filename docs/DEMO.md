# Demo Guide (EN)

## What this demo shows
1) Spin up all services.
2) Load open data via Airflow.
3) Train multiple models and log metrics in MLflow.
4) Set alias `champion` for the best model.
5) Autoserve tracks aliases and serves the `champion` model.
6) Prometheus/Grafana monitor from the start, including model reloads.
7) Loki collects logs from all containers (via Promtail).
8) Run inference via Airflow and send predictions to the champion model.
9) Grafana tracks health, probe latency, and logs.

## Step‑by‑step
1) Start services
```
set -a
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/mlops
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/grafana
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/minio
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/mlflow
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/airflow
set +a
docker compose up -d --build
```

2) Load data (open dataset)
- In Airflow UI run DAG: `dag_data_predictions`.
- This loads the `iris` dataset into the external application database.

3) Train and log
- Run DAG: `dag_training`.
- It trains multiple baseline models (e.g., RandomForest, SVM, Logistic/Ridge) and logs metrics + artifacts into MLflow.

4) Promote best model
- In MLflow UI set alias `champion` for the best run.
- Optionally assign a second version to alias `challenger` for side-by-side validation.
- Ensure Vault config provides `MLFLOW_SERVE_ALIASES` with `champion,challenger`.

5) Verify serving
- `mlflow-autoserve` starts a `mlflow models serve` container per alias.
- Health check is available at `/ping` inside the Docker network.

6) Inference
- Run DAG: `dag_inference`.
- It loads the configured serving alias and stores predictions in the external application database.

7) Observability
- Grafana **Service Health** shows new `mlflow-serve-*` targets via Blackbox.
- Loki logs are available in Grafana (Loki datasource).

Dashboards are provisioned from `monitoring/grafana/dashboards-min`.
