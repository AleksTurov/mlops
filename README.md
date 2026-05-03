# Production ML without CI/CD: Deploy models by switching aliases

Deployment is not a pipeline.

Deployment is a label.

This repository is a self-contained open-source MLOps demo where model rollout happens by switching MLflow aliases such as `champion` and `challenger`. Autoserve watches the registry, recreates the right MLflow Serve containers, and the monitoring stack shows the result immediately.
In this demo, autoserved containers stay inside the Docker network, so the most reliable request path is the one already used in the integration tests.

Use these entry points:
In this demo, autoserved containers stay inside the Docker network, so the canonical request path is the one already covered by the repo tests and helper scripts.

Use these entry points:
- exact end-to-end request path: [test/test_integration_predictions.py](test/test_integration_predictions.py)
- manual request helper: [scripts/predict_request.py](scripts/predict_request.py)
- input schema inspection: [scripts/print_model_input_schema.py](scripts/print_model_input_schema.py)

To replay the tested champion request path directly:

```bash
RUN_INTEGRATION_TESTS=1 .venv/bin/python -m pytest -q test/test_integration_predictions.py -k champion
```

The current demo payload source is `data_contract/sample_input.csv`, logged during training and reused by the integration test.
If you prefer raw Docker commands:

```bash
docker compose up -d --build
./scripts/run_demo_checks.sh
```

Optional customization:

```bash
cp .env.example .env
make demo
```
## What you get in 2 minutes

- trained model in MLflow
- champion/challenger aliases
- auto-deployed serving containers
- Grafana dashboards with live health

## Why Not Traditional Deployment?

| Problem | Traditional workflow | This project |
|---|---|---|
| Deployment trigger | CI/CD pipeline | MLflow alias switch |
| Rollback | Manual redeploy | Repoint alias |
| Serving control | External service layer | Registry-driven autoserve |
| Demo reproducibility | Usually fragmented | One local stack |
| Observability linkage | Added later | Built into the serving path |

## Why This Instead of Full MLOps Platforms?

This project is intentionally narrower than platforms like Kubeflow, SageMaker, or other full-stack MLOps suites.

It is useful when you want:
- no cloud dependency for the core demo path
- no vendor lock-in around deployment semantics
- a stack you can reproduce locally
- minimal setup focused on model registry workflows
- a clear alias-driven rollout story instead of a broad platform installation

It is not trying to replace every feature of a full MLOps platform. It is a smaller, easier-to-explain system for teams validating registry-driven deployment and rollback.

## Who Is This For?

- ML engineers building internal platforms
- data scientists moving models closer to production
- teams experimenting with model registry workflows
- engineering teams that need a reproducible local demo for model rollout patterns

## Use Cases

This architecture is designed for:
- champion/challenger rollouts with MLflow Model Registry
- production-style demos for online and offline inference flows
- internal platform teams validating registry-driven deployment patterns
- scoring systems where rollback speed matters more than release ceremony

## What Starts Automatically

After `make demo`, the stack bootstraps:
- MinIO bucket creation for MLflow artifacts
- Airflow metadata initialization and demo admin user
- `dag_data_predictions` and `dag_training`
- alias-driven autoserve for `champion` and `challenger`
- a prediction integration path that writes explicit MLflow traces
- Grafana, Prometheus, Loki, Promtail, and Blackbox monitoring

Airflow is the default orchestration path in this demo, but the alias-driven deployment model does not depend on Airflow specifically. The same registry and autoserve flow can be fed from notebook-driven experiments or other training pipelines.

The public repo does not use Vault and runs under the isolated Compose project `mlops-demo` with network `mlops-demo_default`.

## API Entry Point

Each deployed alias is exposed by an MLflow Serve container with:
- health endpoint: `GET /ping`
- inference endpoint: `POST /invocations`

In this demo, autoserved containers stay inside the Docker network. The repo already includes the request path used in integration checks:

```bash
docker ps --format '{{.Names}}' | grep mlflow-serve-
python scripts/predict_request.py --url http://<serve-container-ip>:8080 --payload /path/to/payload.json
```

If you want a direct demo request with `curl`, resolve the running champion container and send one sample Iris row:

```bash
SERVE_URL="http://$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlflow-serve-iris_classifier_iris-champion):8080"

curl -X POST "$SERVE_URL/invocations" \
    -H "Content-Type: application/json" \
    -d '{"dataframe_records":[{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}]}'
```

If you prefer the built-in Python helper, use the same endpoint with a temporary payload file:

```bash
SERVE_URL="http://$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlflow-serve-iris_classifier_iris-champion):8080"
PAYLOAD_FILE="$(mktemp)"

printf '%s\n' '{"dataframe_records":[{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}]}' > "$PAYLOAD_FILE"

python scripts/predict_request.py --url "$SERVE_URL" --payload "$PAYLOAD_FILE"
```

To inspect the expected input format for a promoted alias:

```bash
python scripts/print_model_input_schema.py --model-name iris_classifier_iris --alias champion --tracking-uri http://localhost:15000
```

The current demo payload source is `data_contract/sample_input.csv`, logged during training and reused by the integration test.

## What To Open After Startup

- MLflow experiment `iris-classification_iris` for runs, models, and traces
- Grafana dashboards `MLOps Overview`, `Service Health Detailed`, and `MLflow Serving`, backed by Prometheus metrics and Loki logs
- Airflow DAGs `dag_data_predictions` and `dag_training`

## Stack Components

- MLflow + PostgreSQL for tracking and model registry
- MinIO for artifacts
- Airflow + PostgreSQL for orchestration and bootstrap
- App PostgreSQL for demo data and predictions
- MLflow autoserve for alias-driven model serving in online and offline-oriented workflows
- Prometheus + Grafana + Loki + Promtail for observability

## Docs

- Documentation hub: [docs/README.md](docs/README.md)
- Simple first-screen diagram: [docs/SIMPLE_DIAGRAM.md](docs/SIMPLE_DIAGRAM.md)
- Conference demo script: [docs/CONFERENCE_SCRIPT.md](docs/CONFERENCE_SCRIPT.md)
- Demo runbook (EN): [docs/DEMO.md](docs/DEMO.md)
- Architecture and system flow (EN): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Architecture and selling points (RU): [docs/ARCHITECTURE_RU.md](docs/ARCHITECTURE_RU.md)
- Python toolkit docs: [README_library.md](README_library.md)

## Notes

- Online inference is handled by alias-driven MLflow Serve containers.
- `GET /metrics` returning `404` on serve containers is expected; health is tracked through `/ping` via Blackbox.

