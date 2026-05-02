import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests


def _is_enabled() -> bool:
    return os.getenv("RUN_INTEGRATION_TESTS", "").lower() in {"1", "true", "yes"}


def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI_TEST", "http://localhost:15000")


def _experiment_name() -> str:
    return os.getenv("MLFLOW_DEMO_EXPERIMENT_NAME", "iris-classification_iris")


def _sanitize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9_.-]", "-", value)
    return value[:63]


def _in_container() -> bool:
    return os.getenv("PREDICTION_TEST_IN_CONTAINER", "").lower() in {"1", "true", "yes"}


def _run_python(code: str, *args: str) -> str:
    root = Path(__file__).resolve().parents[1]
    if _in_container():
        completed = subprocess.run(
            [sys.executable, "-c", code, *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    completed = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "airflow-webserver",
            "python",
            "-c",
            code,
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _mlflow_api_get(path: str, **params) -> dict:
    response = requests.get(f"{_tracking_uri().rstrip('/')}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _resolve_model_version(alias: str):
    code = """
import json
import sys
import time
from mlflow.tracking import MlflowClient

alias = sys.argv[1]
experiment_name = sys.argv[2]
client = MlflowClient(tracking_uri='http://mlflow:5000')
deadline = time.time() + 180
while time.time() < deadline:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        for model in client.search_registered_models():
            try:
                mv = client.get_model_version_by_alias(model.name, alias)
            except Exception:
                continue
            run = client.get_run(mv.run_id)
            if run.info.experiment_id == experiment.experiment_id:
                print(json.dumps({'model_name': model.name, 'model_version': {'version': mv.version, 'run_id': mv.run_id}}))
                raise SystemExit(0)
    time.sleep(3)

raise SystemExit(f'No model version found for alias={alias} in experiment={experiment_name}')
"""
    payload = json.loads(_run_python(code, alias, _experiment_name()))
    return payload["model_name"], payload["model_version"]


def _serving_payload(run_id: str) -> dict:
    code = """
import json
import pandas as pd
import sys
import tempfile
from mlflow.tracking import MlflowClient

run_id = sys.argv[1]
client = MlflowClient(tracking_uri='http://mlflow:5000')
with tempfile.TemporaryDirectory() as tmpdir:
    path = client.download_artifacts(run_id, 'data_contract/sample_input.csv', dst_path=tmpdir)
    frame = pd.read_csv(path).head(1)
    print(json.dumps({'dataframe_records': frame.to_dict(orient='records')}))
"""
    return json.loads(_run_python(code, run_id))


def _container_endpoint(model_name: str, alias: str) -> str:
    container_name = _sanitize_name(f"mlflow-serve-{model_name}-{alias}")
    if _in_container():
        return f"http://{container_name}:8080"

    inspect = subprocess.run(
        ["docker", "inspect", container_name],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(inspect.stdout)[0]
    labels = payload.get("Config", {}).get("Labels", {})
    networks = payload.get("NetworkSettings", {}).get("Networks", {})
    assert networks, f"No networks found for container {container_name}"

    network_info = next(iter(networks.values()))
    ip_address = network_info.get("IPAddress")
    assert ip_address, f"No IP address found for container {container_name}"

    port = labels.get("mlflow_port", "8080")
    return f"http://{ip_address}:{port}"


def _emit_trace(alias: str, model_name: str, response_body: dict) -> int:
    code = """
import json
import mlflow
import sys
from mlflow.tracking import MlflowClient

experiment_name = sys.argv[1]
alias = sys.argv[2]
model_name = sys.argv[3]
response_body = json.loads(sys.argv[4])

mlflow.set_tracking_uri('http://mlflow:5000')
mlflow.set_experiment(experiment_name)
with mlflow.start_span(
    name=f'integration-prediction-{alias}',
    span_type='TEST',
    attributes={
        'alias': alias,
        'model_name': model_name,
        'response_preview': json.dumps(response_body)[:500],
    },
):
    pass

client = MlflowClient(tracking_uri='http://mlflow:5000')
experiment = client.get_experiment_by_name(experiment_name)
traces = list(client.search_traces(experiment_ids=[experiment.experiment_id], max_results=20))
print(len(traces))
"""
    return int(_run_python(code, _experiment_name(), alias, model_name, json.dumps(response_body)))


@pytest.mark.parametrize("alias", ["champion", "challenger"])
@pytest.mark.skipif(not _is_enabled(), reason="Set RUN_INTEGRATION_TESTS=1 to enable")
def test_predictions_from_serving_container_by_alias_and_experiment(alias: str, tmp_path: Path):
    del tmp_path
    model_name, model_version = _resolve_model_version(alias)
    endpoint = _container_endpoint(model_name, alias)
    payload = _serving_payload(model_version["run_id"])

    health = requests.get(f"{endpoint}/ping", timeout=15)
    assert health.status_code == 200

    response = requests.post(
        f"{endpoint}/invocations",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    body = response.json()
    assert body is not None
    assert body != {}

    trace_count = _emit_trace(alias, model_name, body)
    assert trace_count >= 1