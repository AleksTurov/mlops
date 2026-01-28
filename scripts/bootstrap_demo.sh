#!/usr/bin/env bash
set -euo pipefail

wait_for() {
  local name="$1"
  local url="$2"
  echo "[bootstrap] waiting for ${name}..."
  until curl -sf "$url" >/dev/null 2>&1; do
    sleep 2
  done
  echo "[bootstrap] ${name} is ready"
}

check_url() {
  local name="$1"
  local url="$2"
  if curl -sf "$url" >/dev/null 2>&1; then
    echo "[check] ${name}: OK (${url})"
  else
    echo "[check] ${name}: FAIL (${url})"
  fi
}

wait_for "MLflow" "http://mlflow:5000/health"
wait_for "Airflow" "http://airflow-webserver:8080/health"

if [[ "${BOOTSTRAP_RESET_MLFLOW:-false}" == "true" ]]; then
  echo "[bootstrap] cleaning MLflow experiments (soft delete, except Default)"
  python - <<'PY'
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()
for exp in client.search_experiments(view_type=ViewType.ALL):
    if exp.name == "Default":
        continue
    client.delete_experiment(exp.experiment_id)
print("[bootstrap] MLflow experiments deleted")
PY
fi

echo "[bootstrap] unpause and trigger DAGs"
airflow dags unpause dag_data_predictions || true
airflow dags unpause dag_training || true

airflow dags trigger dag_data_predictions || true
airflow dags trigger dag_training || true

check_url "MLflow" "http://mlflow:5000/health"
check_url "Airflow" "http://airflow-webserver:8080/health"
check_url "MinIO" "http://minio:9000/minio/health/live"
check_url "Prometheus" "http://prometheus:9090/-/healthy"
check_url "Loki" "http://loki:3100/ready"
check_url "Grafana" "http://grafana:3000/api/health"

cat <<EOF

[bootstrap] UI links:
- MLflow:    http://localhost:${MLFLOW_PORT}
- Airflow:   http://localhost:${AIRFLOW_WEB_PORT}
- MinIO:     http://localhost:${MINIO_CONSOLE_PORT}
- Grafana:   http://localhost:${GRAFANA_PORT}
- Prometheus:http://localhost:${PROMETHEUS_PORT}

[bootstrap] Next steps:
- В MLflow проверьте зарегистрированные модели и alias ${MLFLOW_MODEL_ALIAS:-Production}.
- В Grafana откройте дашборд Service Health Detailed.
EOF
