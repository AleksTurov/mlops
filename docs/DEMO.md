# Demo Guide (EN)

## What this demo shows
1) Spin up all services.
2) Load open data via Airflow.
3) Train multiple models and log metrics in MLflow.
4) Set alias `Production` for the best model.
5) Model server tracks alias and serves the `Production` model.
6) Prometheus/Grafana monitor from the start, including model reloads.
7) Run inference via Airflow and send predictions to the production model.
8) Grafana tracks requests/latency/errors.

## Step‑by‑step
1) Start services
```
cp env.dev.example .env
docker compose --env-file .env up -d --build
```

2) Load data (open dataset)
- In Airflow UI run DAG: `dag_data_predictions`.
- This loads the `iris` dataset into `app-db`.

3) Train and log
- Run DAG: `dag_training`.
- It trains multiple baseline models (e.g., RandomForest, SVM, Logistic/Ridge) and logs metrics + artifacts into MLflow.

4) Promote best model
- In MLflow UI set alias `Production` for the best run.
- Ensure `.env` has `MODEL_ALIAS=Production` (default).

5) Verify serving
- `model-server` auto-loads `Production` alias.
- `GET /model/info` returns model name/version/alias.

6) Inference
- Run DAG: `dag_inference`.
- It loads `Production` alias and stores predictions in app-db.

7) Observability
- Grafana dashboards show request latency, RPS, errors.
- Tag watcher shows current alias/version.
