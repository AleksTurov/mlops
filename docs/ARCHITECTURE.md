# Architecture and Operating Model (EN)

This file explains how the stack is structured, why the alias-driven deployment model matters, and how the runtime behaves once the demo is up. For startup and verification commands, use [docs/DEMO.md](docs/DEMO.md).

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

That changes the operating model in an important way: promotion and rollback move from infrastructure choreography to registry metadata.

## 3) End-to-End Flow
1. Airflow loads demo data.
2. Airflow trains multiple candidates.
3. The best model is logged to MLflow and registered.
4. MLflow aliases point to the active model versions.
5. Autoserve notices alias changes and recreates `mlflow-serve-*` containers.
6. Prometheus and Grafana show health for both base services and alias endpoints.
7. Bootstrap runs a prediction integration test, which also writes explicit MLflow traces.

The result is a closed loop: train, register, promote, serve, and observe all happen inside one reproducible local stack.

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
- **Dashboards**: Grafana exposes `MLOps Overview`, `Service Health Detailed`, and `MLflow Serving` for the public demo.

The observability layer is intentionally close to the deployment mechanism: when an alias changes, the same stack shows target health, logs, and trace evidence without requiring a separate platform.

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
- the registry UI shows the decision point clearly,
- the serving target changes immediately after the alias move,
- the dashboards reflect the rollout without a separate release step,
- rollback is the same operation in reverse.

## 8) Traditional vs This Approach

| Step | Traditional | This project |
|---|---|---|
| Deployment | CI/CD | Alias switch |
| Rollback | Manual | Instant |
| Serving | Custom API | MLflow serve |
| Release target | Environment | Registry alias |
| Validation | Separate process | `challenger` alias |

In other words, this architecture does not remove operational discipline. It compresses the path from model decision to serving decision so the rollout mechanism is simpler, faster, and easier to explain.
