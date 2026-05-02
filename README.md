# MLOps Platform: Alias-Driven Deployment with MLflow

![Architecture](docs/Mlops_01.png)

This repository is a self-contained public demo of an alias-driven MLOps stack built entirely from open-source components.

It starts a full stack with one command:

```bash
docker compose up -d --build
```

The stack includes:
- MLflow + PostgreSQL for tracking and model registry
- MinIO for artifacts
- Airflow + PostgreSQL for orchestration and demo bootstrap
- App PostgreSQL for demo data and predictions
- MLflow autoserve for alias-driven online serving
- Prometheus + Grafana + Loki + Promtail for observability

## Core Idea

Deployment is not a pipeline.

Deployment is a label.

By switching MLflow aliases, this stack can:
- deploy a new model instantly
- avoid downtime during rollout
- roll back in seconds
- keep serving decisions inside the model registry instead of an external release workflow

## Simple Flow

```mermaid
flowchart LR
    A[Train in Airflow] --> B[Register model in MLflow]
    B --> C[Assign alias champion or challenger]
    C --> D[Autoserve detects alias change]
    D --> E[MLflow Serve container is recreated]
    E --> F[Prometheus and Grafana show health]
```

## Quick Start

```bash
docker compose up -d --build
```

After the stack is up, run the one-command verification:

```bash
./scripts/run_demo_checks.sh
```

Optional customization:

```bash
cp .env.example .env
docker compose up -d --build
```

Default demo credentials:
- Airflow UI: `admin` / `admin`
- Grafana: `admin` / `admin`
- MinIO: `minioadmin` / `minioadmin123`

Default demo ports:
- MLflow UI: `http://localhost:15000`
- Airflow UI: `http://localhost:18885`
- MinIO API: `http://localhost:19000`
- MinIO Console: `http://localhost:19023`
- Grafana: `http://localhost:13000`
- Prometheus: `http://localhost:19090`
- Loki: `http://localhost:13100`
- Autoserve health: `http://localhost:15010/health`

The public repo does not use Vault and runs under the isolated Compose project `mlops-demo` with network `mlops-demo_default`.

## What Starts Automatically

- MinIO bucket bootstrap
- Airflow metadata initialization
- Demo admin user creation in Airflow
- `dag_data_predictions` and `dag_training`
- alias-driven autoserve for `champion` and `challenger`
- prediction integration test from bootstrap
- explicit MLflow traces for the demo experiment

## Why This Architecture Works

- It keeps deployment semantics inside the model registry, where model decisions already live.
- It avoids a separate serving control plane for the common rollout path.
- It turns rollback into a metadata change instead of an infrastructure event.
- It gives one reproducible stack for demos, development, and debugging.
- It keeps health, logs, and traces close to the serving path that operators actually care about.

## After Startup

Run the one-command verification:

```bash
./scripts/run_demo_checks.sh
```

Then use these entry points:
- MLflow experiment `iris-classification_iris` for runs, models, and traces
- Grafana dashboards `MLOps Overview`, `Service Health Detailed`, and `MLflow Serving`
- Airflow DAGs `dag_data_predictions` and `dag_training`

The project is organized so that each document has one job:
- [docs/DEMO.md](docs/DEMO.md) is the runbook for live walkthroughs
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains the system design and operating model
- [docs/ARCHITECTURE_RU.md](docs/ARCHITECTURE_RU.md) presents the same architecture in Russian for conference and stakeholder context

## Docs

- Demo runbook (EN): [docs/DEMO.md](docs/DEMO.md)
- Architecture and system flow (EN): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Architecture and selling points (RU): [docs/ARCHITECTURE_RU.md](docs/ARCHITECTURE_RU.md)
- Python toolkit docs: [README_library.md](README_library.md)

## Notes

- Online inference is handled by alias-driven MLflow Serve containers.
- `GET /metrics` returning `404` on serve containers is expected; health is tracked through `/ping` via Blackbox.
- The current demo payload source is `data_contract/sample_input.csv`, which is logged during training and reused by the integration test.

