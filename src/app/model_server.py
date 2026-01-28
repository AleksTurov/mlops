import os
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Thread
from time import sleep
from typing import List, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import get_settings
from core.logger import logger
from db.crud import get_latest_dataset
from db.session import SessionLocal, init_db


class PredictRequest(BaseModel):
    records: List[dict]


class ModelState:
    def __init__(self) -> None:
        self.model = None
        self.model_name: Optional[str] = None
        self.model_version: Optional[str] = None
        self.alias: Optional[str] = None
        self.role: Optional[str] = None


state = ModelState()
model_cache: dict[str, "ModelCacheEntry"] = {}


@dataclass
class ModelCacheEntry:
    model: object
    model_name: str
    model_version: str
    alias: str
    role: Optional[str]
    loaded_at: datetime


def _set_mlflow_env(settings) -> None:
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)


def _normalize_alias(environment: str) -> str:
    env = environment.strip().lower()
    if env in {"prod", "production"}:
        return "Production"
    if env in {"stage", "staging"}:
        return "Staging"
    if env in {"dev", "development"}:
        return "Development"
    return environment


def _load_model_by_alias(model_name: str, alias: str) -> ModelCacheEntry:
    settings = get_settings()
    _set_mlflow_env(settings)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        version = client.get_model_version_by_alias(model_name, alias).version
    except mlflow.exceptions.RestException as exc:
        raise HTTPException(status_code=404, detail=f"Alias not found: {model_name}@{alias}") from exc

    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)

    return ModelCacheEntry(
        model=model,
        model_name=model_name,
        model_version=str(version),
        alias=alias,
        role=alias.lower(),
        loaded_at=datetime.now(timezone.utc),
    )


def _get_cached_model(model_name: str, alias: str, force_reload: bool = False) -> ModelCacheEntry:
    key = f"{model_name}@{alias}"
    if not force_reload and key in model_cache:
        return model_cache[key]
    entry = _load_model_by_alias(model_name, alias)
    model_cache[key] = entry
    return entry


def _resolve_model_name(settings) -> str:
    if settings.model_name_override:
        return settings.model_name_override
    init_db()
    with SessionLocal() as db:
        dataset = get_latest_dataset(db)
    if dataset is None:
        return settings.mlflow_model_name
    return f"{settings.mlflow_model_name}_{dataset.name}"


def _load_model() -> None:
    settings = get_settings()
    _set_mlflow_env(settings)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    model_name = _resolve_model_name(settings)
    alias = settings.model_alias

    version = client.get_model_version_by_alias(model_name, alias).version
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)

    state.model = model
    state.model_name = model_name
    state.model_version = str(version)
    state.alias = alias
    state.role = settings.model_role

    logger.info("Loaded model %s@%s (v%s)", model_name, alias, version)


def _poll_model() -> None:
    settings = get_settings()
    while True:
        try:
            _load_model()
        except Exception as exc:
            logger.warning("Model reload failed: %s", exc)
        sleep(settings.model_server_poll_seconds)


app = FastAPI(title="MLflow Model Server", version="1.0.0")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.on_event("startup")
def startup() -> None:
    try:
        _load_model()
    except Exception as exc:
        logger.warning("Model load skipped on startup: %s", exc)
    Thread(target=_poll_model, daemon=True).start()
    logger.info("Model server started")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": state.model_name, "alias": state.alias, "version": state.model_version}


@app.get("/model/info")
def model_info() -> dict:
    return {
        "model_name": state.model_name,
        "alias": state.alias,
        "version": state.model_version,
        "role": state.role,
    }


@app.get("/environment/{environment}/{experiment}/model/info")
def model_info_dynamic(environment: str, experiment: str, refresh: bool = False) -> dict:
    settings = get_settings()
    alias = _normalize_alias(environment)
    model_name = f"{settings.mlflow_model_name}_{experiment}"
    entry = _get_cached_model(model_name, alias, force_reload=refresh)
    return {
        "model_name": entry.model_name,
        "alias": entry.alias,
        "version": entry.model_version,
        "role": entry.role,
        "loaded_at": entry.loaded_at.isoformat(),
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    if not payload.records:
        raise HTTPException(status_code=400, detail="Empty records")
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame(payload.records)
    expected = getattr(state.model, "feature_names_in_", None)
    if expected is not None:
        missing = [col for col in expected if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        df = df.loc[:, list(expected)]
    preds = state.model.predict(df)
    result = {"predictions": preds.tolist(), "model": state.model_name, "alias": state.alias}

    if hasattr(state.model, "predict_proba"):
        probas = state.model.predict_proba(df)
        result["probabilities"] = probas.tolist()

    return result


@app.post("/environment/{environment}/{experiment}/predict")
def predict_dynamic(
    environment: str,
    experiment: str,
    payload: PredictRequest,
    refresh: bool = False,
) -> dict:
    if not payload.records:
        raise HTTPException(status_code=400, detail="Empty records")

    settings = get_settings()
    alias = _normalize_alias(environment)
    model_name = f"{settings.mlflow_model_name}_{experiment}"
    entry = _get_cached_model(model_name, alias, force_reload=refresh)

    df = pd.DataFrame(payload.records)
    expected = getattr(entry.model, "feature_names_in_", None)
    if expected is not None:
        missing = [col for col in expected if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        df = df.loc[:, list(expected)]

    preds = entry.model.predict(df)
    result = {
        "predictions": preds.tolist(),
        "model": entry.model_name,
        "alias": entry.alias,
    }

    if hasattr(entry.model, "predict_proba"):
        probas = entry.model.predict_proba(df)
        result["probabilities"] = probas.tolist()

    return result
