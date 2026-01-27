import os
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


def _set_mlflow_env(settings) -> None:
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)


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


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    if not payload.records:
        raise HTTPException(status_code=400, detail="Empty records")
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame(payload.records)
    preds = state.model.predict(df)
    result = {"predictions": preds.tolist(), "model": state.model_name, "alias": state.alias}

    if hasattr(state.model, "predict_proba"):
        probas = state.model.predict_proba(df)
        result["probabilities"] = probas.tolist()

    return result
