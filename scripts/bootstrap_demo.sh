#!/usr/bin/env sh
set -eu

http_ok() {
  python - "$1" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        sys.exit(0 if response.status < 400 else 1)
except Exception:
    sys.exit(1)
PY
}

wait_for() {
  name="$1"
  url="$2"
  echo "[bootstrap] waiting for ${name}..."
  until http_ok "$url"; do
    sleep 2
  done
  echo "[bootstrap] ${name} is ready"
}

check_url() {
  name="$1"
  url="$2"
  if http_ok "$url"; then
    echo "[check] ${name}: OK (${url})"
  else
    echo "[check] ${name}: FAIL (${url})"
  fi
}

run_prediction_trace_test() {
  attempts="${BOOTSTRAP_TEST_RETRIES:-20}"
  delay_seconds="${BOOTSTRAP_TEST_RETRY_DELAY_SECONDS:-5}"
  attempt=1

  echo "[bootstrap] running prediction integration test to generate traces"
  while [ "$attempt" -le "$attempts" ]; do
    if RUN_INTEGRATION_TESTS=1 PREDICTION_TEST_IN_CONTAINER=1 python -m pytest -q /opt/airflow/test/test_integration_predictions.py; then
      echo "[bootstrap] prediction integration test passed"
      return 0
    fi

    echo "[bootstrap] prediction integration test failed on attempt ${attempt}/${attempts}; retrying in ${delay_seconds}s"
    attempt=$((attempt + 1))
    sleep "$delay_seconds"
  done

  echo "[bootstrap] prediction integration test failed after ${attempts} attempts"
  return 1
}

wait_for "MLflow" "http://mlflow:5000/health"
wait_for "Airflow" "http://airflow-webserver:8080/health"

if [ "${BOOTSTRAP_RESET_MLFLOW:-false}" = "true" ]; then
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

run_prediction_trace_test

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
- В MLflow проверьте зарегистрированные модели и alias ${MLFLOW_MODEL_ALIAS:-champion}.
- В MLflow откройте experiment iris-classification_iris и вкладку Traces после bootstrap test run.
- Для проверки кандидата переведите новую версию в alias challenger.
- В Grafana откройте дашборд Service Health Detailed.
EOF
