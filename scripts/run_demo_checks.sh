#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[checks] Python environment not found at $PYTHON_BIN" >&2
  echo "[checks] Configure the workspace virtualenv first or set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

check_url() {
  name="$1"
  url="$2"
  "$PYTHON_BIN" - "$name" "$url" <<'PY'
import sys
import urllib.request

name, url = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read(120).decode("utf-8", "replace").strip().replace("\n", " ")
        print(f"[smoke] {name}: {response.status} {url} {body}")
except Exception as exc:
    print(f"[smoke] {name}: ERROR {url} {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

echo "[checks] smoke-checking user-facing endpoints"
check_url "MLflow" "${MLFLOW_HEALTH_URL:-http://localhost:15000/health}"
check_url "Airflow" "${AIRFLOW_HEALTH_URL:-http://localhost:18885/health}"
check_url "Grafana" "${GRAFANA_HEALTH_URL:-http://localhost:13000/api/health}"
check_url "Prometheus" "${PROMETHEUS_HEALTH_URL:-http://localhost:19090/-/healthy}"
check_url "Loki" "${LOKI_HEALTH_URL:-http://localhost:13100/ready}"
check_url "MinIO" "${MINIO_HEALTH_URL:-http://localhost:19000/minio/health/live}"

echo "[checks] running full pytest suite with integration checks enabled"
cd "$ROOT_DIR"
RUN_INTEGRATION_TESTS=1 "$PYTHON_BIN" -m pytest -q "$@"