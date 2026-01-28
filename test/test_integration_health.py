import os
import requests
import pytest


def _is_enabled() -> bool:
    return os.getenv("RUN_INTEGRATION_TESTS", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _is_enabled(), reason="Set RUN_INTEGRATION_TESTS=1 to enable")
def test_mlflow_health():
    url = os.getenv("MLFLOW_HEALTH_URL", "http://localhost:5001/health")
    resp = requests.get(url, timeout=10)
    assert resp.status_code == 200


@pytest.mark.skipif(not _is_enabled(), reason="Set RUN_INTEGRATION_TESTS=1 to enable")
def test_grafana_health():
    url = os.getenv("GRAFANA_HEALTH_URL", "http://localhost:3001/api/health")
    resp = requests.get(url, timeout=10)
    assert resp.status_code == 200
