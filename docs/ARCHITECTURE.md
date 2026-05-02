# Architecture and System Flow (EN)

## 1) Service roles
- **MLflow** — model registry and experiment tracking.
- **PostgreSQL (mlflow-db)** — MLflow metadata.
- **MinIO** — model artifacts (S3-compatible).
- **Application PostgreSQL (external)** — app-side data and predictions (provided via Vault env variables).
- **MLflow Autoserve** — watcher that starts MLflow Serve for each alias.
- **MLflow Serve containers** — online inference per model+alias.
- **Prometheus/Grafana** — health/metrics dashboards.
- **Loki/Promtail** — log aggregation and search in Grafana.
- **Blackbox exporter** — HTTP health probes.

## 2) Data flow (end-to-end)
1. **Experimenting** → runs are logged to MLflow; artifacts are stored in MinIO.
2. **Registry & aliases** → model versions are managed by aliases (`champion`, `challenger`) in MLflow.
3. **Serving** → `mlflow-autoserve` detects aliases and starts `mlflow models serve` containers.
4. **Observability** → Blackbox checks `/ping` for each `mlflow-serve-*` container; Grafana shows health status.
5. **Logs** → Promtail ships Docker logs to Loki; Grafana shows logs by service/container.

## 2.1) Current testing target
- Current MLflow experiment for tag/alias validation: **scoring_eldik**.
- Promotion flow: assign `challenger` → validate → assign `champion`.
- Rollback flow: reassign `champion` to previous version.

## 3) Serving flow (MLflow Serve)
- Each alias spawns a dedicated container named like `mlflow-serve-<model>-<alias>`.
- Health endpoint: `GET /ping` (inside Docker network).
- Inference endpoint: `POST /invocations` with MLflow scoring format.
- Expected input schema is primarily stored in MLflow model artifacts: `model/MLmodel` for signature and `model/serving_input_example.json` for a ready scoring payload. `data_contract/input_schema.json` is optional and model-dependent.
- MLflow Serve does **not** expose Prometheus `/metrics`; use Blackbox health probes for availability.

## 3.1) Checklist for Multi-Input PyTorch in MLflow
- Do not rely on plain `mlflow.pytorch.log_model()` when the model `forward()` expects multiple arguments like `x_num, x_cat, lengths`.
- Wrap the model in a custom `mlflow.pyfunc.PythonModel` whose `predict()` accepts the serving payload and explicitly constructs all input tensors.
- Make the `predict()` contract match the future REST contract exactly. If `/invocations` will receive JSON with `x_num`, `x_cat`, and `lengths`, the wrapper should read those exact keys.
- Keep `signature`, `input_example`, and `serving_input_example` consistent with each other. They must describe the same shapes, dtypes, and field names.
- Prefer logging a dict-shaped input example that mirrors the real serving payload rather than a convenience example that only worked in notebooks.
- Validate the wrapper locally before registration by calling `loaded_model.predict(...)` with the exact payload shape planned for `/invocations`.
- After logging, inspect `model/MLmodel` and confirm that the signature matches the model version you just registered.
- After logging, inspect `model/serving_input_example.json` and use it as the first smoke-test payload.
- Register the model version only after the smoke test succeeds through `mlflow models serve` or the same docker-image serving path used in production.
- If model inputs change between versions, regenerate signature and examples for every version. Never reuse old examples from a previous architecture.
- If the model cannot be represented cleanly as a single tensor or DataFrame, treat a custom pyfunc wrapper as required, not optional.
- Promotion rule: move alias to `champion` only after `/ping` is healthy and `/invocations` succeeds with the stored serving example.

## 4) Observability
- **Service health**: `probe_success` from Blackbox.
- **Dashboards**: provisioned from `monitoring/grafana/dashboards-min`.
	- **Service Health Detailed** (all services + model/alias status)
	- **MLflow Serving** (model alias health and probe latency)
- **Logs**: Loki via Promtail; filter by `container` or `service` labels.

## 5) Databases and network
- The stack runs a local Postgres only for MLflow metadata (`mlflow-db`).
- Application data DB is external and injected through Vault variables (`APP_DB_*` / `APP_DB_URL`).
- This separation keeps MLflow metadata independent from Airflow/application operational data.

## 6) Default runtime conventions
- Default aliases: `champion`, `challenger`.
- Default serving projects: `models_champion`, `models_challenger`.
- The runtime no longer includes a separate legacy model-server service; online inference is handled only by alias-driven MLflow Serve containers.

## 7) Endpoints
All ports are configured via exported environment variables from Vault (`*_PORT`).
- MLflow UI: http://localhost:${MLFLOW_PORT}
- MinIO Console: http://localhost:${MINIO_CONSOLE_PORT}
- Grafana: http://localhost:${GRAFANA_PORT}
- Prometheus: http://localhost:${PROMETHEUS_PORT}
- Loki: http://localhost:${LOKI_PORT}
