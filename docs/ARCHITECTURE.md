# Architecture and System Flow (EN)

## 1) Service Roles
- **MLflow**: experiment tracking, model registry, aliases, traces.
- **PostgreSQL (`mlflow-db`)**: MLflow metadata.
- **MinIO**: model artifacts.
- **Airflow + PostgreSQL (`airflow-db`)**: orchestration, bootstrap, demo jobs.
- **Application PostgreSQL (`app-db`)**: demo data and prediction results.
- **MLflow Autoserve**: reconciles aliases to serving containers.
- **MLflow Serve containers**: one online inference endpoint per `model@alias`.
- **Prometheus + Blackbox exporter**: service and endpoint health.
- **Grafana**: dashboards and trace/log navigation.
- **Loki + Promtail**: container logs.

## 2) Core Innovation

Deployment is not a pipeline.

Deployment is a label.

The serving target is controlled by MLflow aliases such as `champion` and `challenger`. Changing an alias changes the deployed model version without rebuilding a custom API service.

## 3) End-to-End Flow
1. Airflow loads demo data.
2. Airflow trains multiple candidates.
3. The best model is logged to MLflow and registered.
4. MLflow aliases point to the active model versions.
5. Autoserve notices alias changes and recreates `mlflow-serve-*` containers.
6. Prometheus and Grafana show health for both base services and alias endpoints.
7. Bootstrap runs a prediction integration test, which also writes explicit MLflow traces.

## 4) Serving Flow
- Each alias creates a container named like `mlflow-serve-<model>-<alias>`.
- Health endpoint: `GET /ping`.
- Inference endpoint: `POST /invocations`.
- Autoserve resolves the source experiment from the model version run and passes `MLFLOW_EXPERIMENT_NAME` into the serve container.
- For the current demo, prediction payloads are sourced from `data_contract/sample_input.csv` logged during training.

## 5) Observability
- **Health**: Blackbox probes `GET /ping` for alias endpoints and health endpoints for infrastructure services.
- **Logs**: Promtail ships container logs to Loki.
- **Traces**: the bootstrap prediction test writes explicit MLflow traces into experiment `iris-classification_iris`.

Important detail:
- MLflow Serve does not expose Prometheus `/metrics` by default.
- `GET /metrics -> 404` on a serve container is expected in this project.

## 6) Demo Runtime Conventions
- Default aliases: `champion`, `challenger`.
- Default serving projects: `models_champion`, `models_challenger`.
- Default experiment: `iris-classification_iris`.
- Public demo project name: `mlops-demo`.
- Public demo network: `mlops-demo_default`.

## 7) Demo Scenario For a Live Talk
1. Train model.
2. Assign alias.
3. Watch auto-deploy.

Why this works well on stage:
- the registry UI shows the decision point,
- the serve container changes are visible immediately,
- the dashboards reflect the rollout without any manual redeploy,
- rollback is just another alias move.

## 8) Traditional vs This Approach

| Step | Traditional | This project |
|---|---|---|
| Deployment | CI/CD | Alias switch |
| Rollback | Manual | Instant |
| Serving | Custom API | MLflow serve |
| Release target | Environment | Registry alias |
| Validation | Separate process | `challenger` alias |

## 9) Main Endpoints
- MLflow UI: `http://localhost:15000`
- Airflow UI: `http://localhost:18885`
- MinIO Console: `http://localhost:19023`
- Grafana: `http://localhost:13000`
- Prometheus: `http://localhost:19090`
- Loki: `http://localhost:13100`

## 10) How To Verify The Demo Quickly
1. `docker compose ps`
2. Open Airflow and check `dag_data_predictions` and `dag_training`
3. Open MLflow and inspect aliases for `iris_classifier_iris`
4. Open the `Traces` tab for `iris-classification_iris`
5. Open Grafana dashboard `Service Health Detailed`
6. Run `RUN_INTEGRATION_TESTS=1 pytest -q test/test_integration_predictions.py` if you want to replay the prediction test manually
