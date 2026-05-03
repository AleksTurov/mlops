# Conference Demo Script

This script is designed for a 3-5 minute live walkthrough of the repository after the stack is already running.

## Goal

Show one idea clearly: deployment is an alias switch in MLflow, not a separate CI/CD ceremony.

## Setup Before You Speak

- Run `make demo` and wait for bootstrap to finish.
- Run `make verify` to confirm the demo state.
- Open these browser tabs in advance:
  - MLflow at `http://localhost:15000`
  - Airflow at `http://localhost:18885`
  - Grafana at `http://localhost:13000`
- Keep [SIMPLE_DIAGRAM.md](SIMPLE_DIAGRAM.md) or the README open as the opening slide.

## 3-5 Minute Talk Track

### 0:00-0:40 - Hook

"Most ML systems still treat deployment as a pipeline problem. This project takes a different view: deployment is just switching an alias in MLflow. The registry decides what is live, autoserve reconciles that decision, and the observability stack shows the result immediately."

### 0:40-1:30 - Show The Simple Flow

Point to the simple diagram and say:

"The flow is intentionally small. We train in Airflow, register in MLflow, switch an alias like `champion`, autoserve recreates the matching serving container, and Grafana shows whether the endpoint is healthy."

### 1:30-2:20 - Show MLflow

In MLflow, open the registered model and point to aliases.

Suggested line:

"This is the control plane. I am not deploying through an external release tool here. I am changing registry metadata. That metadata becomes the serving target."

If you want to mention rollback:

"Rollback is the same action in reverse. Move the alias back, and autoserve reconciles again."

### 2:20-3:10 - Show Autoserve Effect And Health

Switch to Grafana and open `Service Health Detailed` or `MLflow Serving`.

Suggested line:

"The point is not only that the alias changed. The point is that the platform shows the resulting runtime state right away: service health, alias endpoints, logs, and traces."

### 3:10-4:00 - Show Airflow Context

Open Airflow and point to `dag_data_predictions` and `dag_training`.

Suggested line:

"Airflow is here for data prep and training orchestration. It is not the deployment mechanism. Deployment happens later, through MLflow aliases."

### 4:00-5:00 - Close

"So the idea behind this demo is simple: remove deployment ceremony from the common ML rollout path. Keep the decision in the model registry, let autoserve apply it, and keep observability next to the serving path."

## Backup Short Version

If you only have 60-90 seconds:

"This stack shows alias-driven deployment with MLflow. Airflow trains, MLflow stores model versions, `champion` and `challenger` aliases decide what should serve, autoserve recreates the serving containers, and Grafana shows health immediately. Rollout and rollback become metadata operations instead of deployment ceremonies."

## Live Demo Cues

- If someone asks where the deployment happened: show the model alias in MLflow.
- If someone asks how you verify runtime state: show Grafana dashboards.
- If someone asks whether the demo is reproducible: show `make demo` and `make verify`.
- If someone asks about inference entrypoints: mention `POST /invocations` and `GET /ping` on the serve container.