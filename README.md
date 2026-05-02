# MLOps Platform (MLflow + Autoserve + Monitoring)

Architecture (visual)

![Architecture](docs/Mlops_01.png)

![Service Health (Grafana)](docs/Mlops_02.png)

This repository provides a practical MLOps runtime for model registry, model serving, and observability:
- experiments and model versions are managed in MLflow,
- artifacts are stored in MinIO,
- model endpoints are auto-created from MLflow aliases,
- health and logs are visible in Grafana (Prometheus + Loki).

Current components
- MLflow + PostgreSQL (mlflow-db) for tracking and registry
- Airflow + PostgreSQL (airflow-db) for orchestration and demo automation
- Application PostgreSQL (app-db) for demo data and predictions
- MinIO for artifacts
- MLflow autoserve for alias-driven serving containers
- Prometheus + Blackbox Exporter + Grafana for health monitoring
- Loki + Promtail for centralized logs

Docs
- Demo guide (EN): [docs/DEMO.md](docs/DEMO.md)
- Architecture and flow (EN): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Architecture (RU): [docs/ARCHITECTURE_RU.md](docs/ARCHITECTURE_RU.md)

Local Python environment (optional)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick start
```bash
docker compose up -d --build
```

Optional customization
```bash
cp .env.example .env
docker compose up -d --build
```

Default demo credentials
- Airflow UI: `admin` / `admin`
- Grafana: `admin` / `admin`
- MinIO: `minioadmin` / `minioadmin123`

Default demo ports
- MLflow UI: `http://localhost:15000`
- Airflow UI: `http://localhost:18885`
- MinIO API: `http://localhost:19000`
- MinIO Console: `http://localhost:19023`
- Grafana: `http://localhost:13000`
- Prometheus: `http://localhost:19090`
- Loki: `http://localhost:13100`
- Autoserve health: `http://localhost:15010/health`

The public repository is self-contained. It does not require Vault and uses the default Compose project name `mlops-demo` plus the default network `mlops-demo_default`, so it does not collide with an existing local `mlops` deployment.

What starts automatically
- MinIO bucket bootstrap
- Airflow metadata initialization
- demo admin user in Airflow
- demo DAG bootstrap for dataset load and model training
- alias-driven autoserve after model registration

How to override for an internal environment
- Copy `.env.example` to `.env` and change ports, credentials, project name, or image settings.
- If you change `COMPOSE_PROJECT_NAME`, also change `COMPOSE_DEFAULT_NETWORK` and `MLFLOW_SERVE_NETWORK` in `.env` so autoserve and Prometheus continue to discover served containers correctly.

Dashboards
- Grafana dashboards are provisioned from [monitoring/grafana/dashboards-min](monitoring/grafana/dashboards-min)
- Use dashboard "Service Health Detailed" to verify each service is alive.
- Use dashboard "MLflow Serving" to inspect model serving status and /ping health.

Platform workflow
1) Register or update model versions in MLflow.
2) Assign aliases in registry (champion, challenger).
3) Autoserve detects aliases and runs/updates serving containers.
4) Blackbox probes /ping, dashboards show model@alias health.
5) Promote or rollback by moving aliases only (no manual container management).

Serving project mapping
- By default autoserve marks serving containers as separate projects:
	- `champion -> models_champion`
	- `challenger -> models_challenger`
- Configure with `MLFLOW_SERVE_ALIAS_PROJECTS` or per-alias overrides `MLFLOW_SERVE_PROJECT_CHAMPION` / `MLFLOW_SERVE_PROJECT_CHALLENGER`.

MLflow experiment for tag/alias testing
- Existing experiment: scoring_eldik
- Recommended model registry flow:
	1. Pick target registered model version from MLflow UI.
	2. Assign alias challenger for validation traffic.
	3. Validate endpoint and metrics.
	4. Move alias champion when ready.
	5. Rollback by switching champion to previous version.

CLI examples (mlops-toolkit)
```bash
mlops-toolkit alias status --model-name scoring_eldik --aliases champion,challenger
mlops-toolkit alias set --model-name scoring_eldik --version <VERSION> --alias challenger
mlops-toolkit alias set --model-name scoring_eldik --version <VERSION> --alias champion
```

Why this setup
- minimal operational overhead for a small team,
- alias-driven champion/challenger releases and rollback,
- clear observability for service and model endpoint health,
- clone-and-run local demo path.

Python toolkit
Install and use the CLI from [README_library.md](README_library.md) to automate MLflow aliases (Phase 1).

How to call served models (inside Docker network):
```bash
docker ps --format '{{.Names}}' | grep mlflow-serve-
docker run --rm --network mlops_default curlimages/curl:8.5.0 -sS http://<mlflow-serve-container>:8080/ping
```
Inference example:
```bash
docker run --rm --network mlops_default curlimages/curl:8.5.0 -sS \
	-H 'Content-Type: application/json' \
	-d '{"dataframe_records":[{"feature_a":1,"feature_b":2}]}' \
	http://<mlflow-serve-container>:8080/invocations
```
The helper script `scripts/predict_request.py` can be used for the same `/invocations` smoke test.
To inspect what the model accepts on input, use `scripts/print_model_input_schema.py` or open the model artifacts in MLflow UI.
The primary sources are `model/MLmodel` for signature and `model/serving_input_example.json` for a ready `/invocations` payload; `data_contract/input_schema.json` is optional and model-dependent.

How to inspect model input in MLflow
1) Open MLflow UI and find the registered model version behind alias `champion` or `challenger`.
2) Open its artifacts and inspect `model/MLmodel`.
3) Read the `signature` section in `MLmodel` to see input names, dtypes, and shapes.
4) Open `model/serving_input_example.json` to get a ready payload for `POST /invocations`.
5) Optionally open `model/input_example.json` for the raw input example and `data_contract/input_schema.json` if the training pipeline logged it.

How to validate scoring
1) Ensure the serving container exists: `docker ps --format '{{.Names}}' | grep mlflow-serve-`.
2) Check health with `GET /ping`.
3) Send the payload from `model/serving_input_example.json` to `POST /invocations`.
4) For a quick CLI check, use `scripts/predict_request.py --url http://<mlflow-serve-container>:8080 --payload <payload.json>`.
5) If the request succeeds and Prometheus shows `probe_success=1`, the model is both serving and reachable.

If the model is a multi-input PyTorch network, see the checklist in `docs/ARCHITECTURE.md` before registering it in MLflow. The important rule is that `/invocations` should be designed first, and the MLflow pyfunc wrapper, signature, and examples should be logged to match that contract exactly.

Notes
- Online inference is performed by alias-driven MLflow serving containers.
- The public demo path is self-contained and does not require Vault.

