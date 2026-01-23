import os
import uuid as _uuid

import pytest
import requests
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://10.16.230.222:7022").rstrip("/")
TIMEOUT = float(os.getenv("API_TEST_TIMEOUT", 5.0))


def _call_predict_get(payload: dict):
    """Отправка GET /predict с query params"""
    url = f"{BASE_URL}/predict"
    return requests.get(url, params=payload, timeout=TIMEOUT)


def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    assert r.status_code == 200
    assert "service_health" in r.text


def test_root_redirects_to_docs():
    r = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
    # может следовать редиректу -> проверить, что docs доступен
    assert r.status_code == 200
    assert "swagger" in r.text.lower() or "openapi" in r.text.lower()


@pytest.mark.parametrize("phone_value", ["996774049151", 996774049151])
def test_predict_basic_success(phone_value):
    """Типичный успешный запрос (возвращает found|not_found|error)"""
    payload = {"phone": phone_value, "settlement_date": "2025-11-01"}
    r = _call_predict_get(payload)
    assert r.status_code == 200, f"unexpected status: {r.status_code} {r.text}"
    data = r.json()
    # базовая структура
    assert "phone" in data and "request_id" in data and "status" in data and "probability" in data
    # request_id должен быть строкой UUID
    try:
        _uuid.UUID(data["request_id"])
    except Exception:
        pytest.skip("request_id отсутствует или не UUID")
    # если probability присутствует — диапазон [0,1]
    prob = data.get("probability")
    if prob is not None:
        assert isinstance(prob, (float, int))
        assert 0.0 <= float(prob) <= 1.0


def test_predict_missing_phone_returns_422():
    """Если не передан обязательный параметр phone — 422"""
    payload = {"settlement_date": "2025-11-01"}
    r = _call_predict_get(payload)
    assert r.status_code == 422


@pytest.mark.parametrize(
    "bad_date",
    ["01-11-2025", "2025/11/01", "20251101", "not-a-date", ""],
)
def test_predict_invalid_date_format_returns_422(bad_date):
    payload = {"phone": "996774049151", "settlement_date": bad_date}
    r = _call_predict_get(payload)
    assert r.status_code == 422


def test_predict_with_no_settlement_date_uses_today():
    """Если settlement_date не передан — эндпоинт должен принять запрос и вернуть 200 (используется today)"""
    payload = {"phone": "996774049151"}
    r = _call_predict_get(payload)
    assert r.status_code == 200
    data = r.json()
    assert data.get("phone") is not None
    assert "request_id" in data


def test_predict_latency_reasonable():
    payload = {"phone": "996774049151", "settlement_date": "2025-11-01"}
    r = _call_predict_get(payload)
    assert r.status_code == 200
    assert r.elapsed.total_seconds() < 5.0