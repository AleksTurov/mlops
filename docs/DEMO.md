# Demo Guide (EN)

This file is the operator runbook for the public demo: start the stack, verify it, and know where to click during a live walkthrough.

Related docs:
- [README.md](../README.md)
- [SIMPLE_DIAGRAM.md](SIMPLE_DIAGRAM.md)
- [CONFERENCE_SCRIPT.md](CONFERENCE_SCRIPT.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)

## What this demo shows
1) Spin up the local stack.
2) Load demo data and train models.
3) Register model versions in MLflow.
4) Serve the active version by alias.
5) Observe rollout health in Grafana.
6) Replay the tested prediction path.

## Step‑by‑step
1) Start services
```
git clone https://github.com/AleksTurov/mlops.git
cd mlops
cp .env.example .env
make demo
```

If you prefer raw Docker commands
```
git clone https://github.com/AleksTurov/mlops.git
cd mlops
cp .env.example .env
docker compose up -d --build
./scripts/run_demo_checks.sh
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
- Autoserve health: `http://localhost:15010/health`

2) What you should get in a couple of minutes
- trained model in MLflow
- `champion` and `challenger` aliases
- auto-deployed serving containers
- Grafana dashboards with live health
- end-to-end demo traces written to MLflow

3) Load data (open dataset)
- `demo-bootstrap` unpauses and triggers `dag_data_predictions` automatically.
- This loads the `iris` dataset into the local demo application database.

4) Train and log
- `demo-bootstrap` also triggers `dag_training` automatically.
- It trains multiple baseline models (e.g., RandomForest, SVM, Logistic/Ridge) and logs metrics + artifacts into MLflow.

5) Promote best model
- The training flow registers the best model and assigns alias `champion` automatically by default.
- Optionally assign a second version to alias `challenger` for side-by-side validation.
- The default stack already enables `champion,challenger` for autoserve.

6) Verify serving
- `mlflow-autoserve` starts a `mlflow models serve` container per alias.
- Health check is available at `/ping` inside the Docker network.

7) Inference
- Run DAG: `dag_inference`.
- It loads the configured serving alias and stores predictions in the local demo application database.

8) Observability
- Grafana dashboards `MLOps Overview`, `Service Health Detailed`, and `MLflow Serving` show infrastructure health, alias targets, latency, and logs.
- Loki logs are available in Grafana (Loki datasource).

Dashboards are provisioned from `monitoring/grafana/dashboards-min`.

## Quick validation after startup

Run the repo-level verification command:

```bash
make verify
```

This command:
- smoke-checks the main local services
- runs the full pytest suite with integration checks enabled
- confirms the current demo runtime before a live session

If you only want to replay the tested champion request path:

```bash
RUN_INTEGRATION_TESTS=1 .venv/bin/python -m pytest -q test/test_integration_predictions.py -k champion
```

Related API entry points:
- [../test/test_integration_predictions.py](../test/test_integration_predictions.py)
- [scripts/predict_request.py](../scripts/predict_request.py)
- [scripts/print_model_input_schema.py](../scripts/print_model_input_schema.py)

## Live walkthrough

1. Open Airflow and confirm `dag_data_predictions` and `dag_training` succeeded.
2. Open MLflow and inspect the latest run and registered model for `iris_classifier_iris`.
3. In MLflow, verify aliases `champion` and `challenger`.
4. Open the `Traces` tab for experiment `iris-classification_iris`.
5. Open Grafana dashboards `MLOps Overview`, `Service Health Detailed`, and `MLflow Serving`.
6. If you want to demonstrate another serving request, rerun the prediction test above.

For a talk-ready script built on top of this runbook, use [CONFERENCE_SCRIPT.md](CONFERENCE_SCRIPT.md).
