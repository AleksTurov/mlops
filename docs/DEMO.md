# Demo Guide (EN)

This file is the operator runbook for the public demo: start the stack, verify it, and know where to click during a live walkthrough.

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

Default demo entry points
- MLflow UI: `http://localhost:15000`
- Airflow UI: `http://localhost:18885`
- MinIO Console: `http://localhost:19023`
- Grafana: `http://localhost:13000`
- Prometheus: `http://localhost:19090`
- Loki: `http://localhost:13100`
- Autoserve health: `http://localhost:15010/health`

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
- Grafana dashboards `MLOps Overview`, `Service Health Detailed`, and `MLflow Serving` show infrastructure health, alias targets, latency, and logs.
- Loki logs are available in Grafana (Loki datasource).

Dashboards are provisioned from `monitoring/grafana/dashboards-min`.

## Quick validation after startup

Run the repo-level verification command:

```bash
./scripts/run_demo_checks.sh
```

This command:
- smoke-checks the main local services
- runs the full pytest suite with integration checks enabled
- confirms the current demo runtime before a live session

If you only want to replay the trace-producing prediction path:

```bash
RUN_INTEGRATION_TESTS=1 .venv/bin/python -m pytest -q test/test_integration_predictions.py
```

## Live walkthrough

1. Open Airflow and confirm `dag_data_predictions` and `dag_training` succeeded.
2. Open MLflow and inspect the latest run and registered model for `iris_classifier_iris`.
3. In MLflow, verify aliases `champion` and `challenger`.
4. Open the `Traces` tab for experiment `iris-classification_iris`.
5. Open Grafana dashboards `MLOps Overview`, `Service Health Detailed`, and `MLflow Serving`.
6. If you want to demonstrate another serving request, rerun the prediction test above.
