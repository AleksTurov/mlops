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
docker compose up -d --build
```

Optional customization
```
cp .env.example .env
docker compose up -d --build
```

The public demo stack starts without Vault and uses safe default ports that do not overlap with the existing local `mlops` runtime.

Default demo credentials
- Airflow UI: `admin` / `admin`
- Grafana: `admin` / `admin`
- MinIO: `minioadmin` / `minioadmin123`

2) Load data (open dataset)
- `demo-bootstrap` unpauses and triggers `dag_data_predictions` automatically.
- This loads the `iris` dataset into the local demo application database.

3) Train and log
- `demo-bootstrap` also triggers `dag_training` automatically.
- It trains multiple baseline models (e.g., RandomForest, SVM, Logistic/Ridge) and logs metrics + artifacts into MLflow.

4) Promote best model
- The training flow registers the best model and assigns alias `champion` automatically by default.
- Optionally assign a second version to alias `challenger` for side-by-side validation.
- The default stack already enables `champion,challenger` for autoserve.

5) Verify serving
- `mlflow-autoserve` starts a `mlflow models serve` container per alias.
- Health check is available at `/ping` inside the Docker network.

6) Inference
- Run DAG: `dag_inference`.
- It loads the configured serving alias and stores predictions in the local demo application database.

7) Observability
- Grafana **Service Health** shows new `mlflow-serve-*` targets via Blackbox.
- Loki logs are available in Grafana (Loki datasource).

Dashboards are provisioned from `monitoring/grafana/dashboards-min`.
