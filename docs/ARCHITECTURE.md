# Architecture and Operating Model (EN)

This file explains the same story as the main README, but from the system-design side: what the stack contains, why alias-driven deployment matters, and what actually happens after `make demo`.

Related docs:
- [README.md](../README.md)
- [SIMPLE_DIAGRAM.md](SIMPLE_DIAGRAM.md)
- [DEMO.md](DEMO.md)
- [CONFERENCE_SCRIPT.md](CONFERENCE_SCRIPT.md)

## 1) Core Idea

Deployment is not a pipeline.

Deployment is a label.

The serving target is controlled by MLflow aliases such as `champion` and `challenger`. Changing an alias changes the deployed model version without rebuilding a custom API service.

That is the main operating-model change in this project: promotion and rollback move from infrastructure choreography to registry metadata.

## 2) What The Stack Contains
- **MLflow**: experiment tracking, model registry, aliases, traces.
- **PostgreSQL (`mlflow-db`)**: MLflow metadata.
- **MinIO**: model artifacts.
- **Airflow + PostgreSQL (`airflow-db`)**: demo orchestration, bootstrap, and scheduled jobs.
- **Application PostgreSQL (`app-db`)**: demo data and prediction results.
- **MLflow Autoserve**: reconciles aliases to serving containers.
- **MLflow Serve containers**: one serving runtime per `model@alias`, usable in online and offline-oriented flows.
- **Prometheus + Blackbox exporter**: service and endpoint health.
- **Grafana**: dashboards that surface Prometheus metrics and Loki logs.
- **Loki + Promtail**: container logs.

Airflow is the default orchestrator in this demo, but it is not a hard requirement for the architecture itself. Model versions can also come from notebook-driven experimentation or other training pipelines, as long as they are registered in MLflow.

## 3) End-to-End Flow
1. Data is prepared from batch sources, application tables, or feature-store-like inputs.
2. Models are trained in Airflow or notebooks.
3. The best model is logged to MLflow and registered.
4. MLflow aliases point to the active model versions.
5. Autoserve notices alias changes and recreates `mlflow-serve-*` containers.
6. Grafana shows health for both base services and alias endpoints using Prometheus metrics and Loki logs.
7. Bootstrap runs a prediction integration test, which also writes explicit MLflow traces.

The result is a closed loop: train, register, promote, serve, and observe all happen inside one reproducible local stack.

## 4) What Starts Automatically

After `make demo`, the stack bootstraps:
- MinIO bucket creation for MLflow artifacts
- Airflow metadata initialization and demo admin user
- `dag_data_predictions` and `dag_training`
- alias-driven autoserve for `champion` and `challenger`
- a prediction integration path that writes explicit MLflow traces
- Grafana, Prometheus, Loki, Promtail, and Blackbox monitoring

The public repo does not use Vault and runs under the isolated Compose project `mlops-demo` with network `mlops-demo_default`.

## 5) Serving Path
- Each alias creates a container named like `mlflow-serve-<model>-<alias>`.
- Health endpoint: `GET /ping`.
- Inference endpoint: `POST /invocations`.
- Autoserve resolves the source experiment from the model version run and passes `MLFLOW_EXPERIMENT_NAME` into the serve container.
- For the current demo, prediction payloads are sourced from `data_contract/sample_input.csv` logged during training.

Canonical request entry points:
- [../test/test_integration_predictions.py](../test/test_integration_predictions.py)
- [../scripts/predict_request.py](../scripts/predict_request.py)
- [../scripts/print_model_input_schema.py](../scripts/print_model_input_schema.py)

## 6) Observability
- **Health**: Blackbox probes `GET /ping` for alias endpoints and health endpoints for infrastructure services.
- **Metrics**: Prometheus stores infrastructure and probe metrics.
- **Logs**: Promtail ships container logs to Loki.
- **Traces**: the bootstrap prediction test writes explicit MLflow traces into experiment `iris-classification_iris`.
- **Dashboards**: Grafana exposes `MLOps Overview`, `Service Health Detailed`, and `MLflow Serving` using Prometheus and Loki as its main datasources.

The observability layer is intentionally close to the deployment mechanism: when an alias changes, the same stack shows target health, logs, and trace evidence without requiring a separate platform.

Important detail:
- MLflow Serve does not expose Prometheus `/metrics` by default.
- `GET /metrics -> 404` on a serve container is expected in this project.

## 7) Demo Runtime Conventions
- Default aliases: `champion`, `challenger`.
- Default serving projects: `models_champion`, `models_challenger`.
- Default experiment: `iris-classification_iris`.
- Public demo project name: `mlops-demo`.
- Public demo network: `mlops-demo_default`.

## 8) Why This Works Well In A Demo
1. Train model.
2. Assign alias.
3. Watch auto-deploy.

Why this works well on stage:
- the registry UI shows the decision point clearly,
- the serving target changes immediately after the alias move,
- the dashboards reflect the rollout without a separate release step,
- rollback is the same operation in reverse.

## 9) Traditional vs This Approach

| Step | Traditional | This project |
|---|---|---|
| Deployment | CI/CD | Alias switch |
| Rollback | Manual | Instant |
| Serving | Custom API | MLflow serve |
| Release target | Environment | Registry alias |
| Validation | Separate process | `challenger` alias |

In other words, this architecture does not remove operational discipline. It compresses the path from model decision to serving decision so the rollout mechanism is simpler, faster, and easier to explain.

## 10) Where To Go Next

- Use [../README.md](../README.md) for quick start and project positioning.
- Use [DEMO.md](DEMO.md) for the startup and validation runbook.
- Use [CONFERENCE_SCRIPT.md](CONFERENCE_SCRIPT.md) for a short stage-friendly walkthrough.
- Use [SCRIPTS.md](SCRIPTS.md) for helper scripts and DAG behavior.
