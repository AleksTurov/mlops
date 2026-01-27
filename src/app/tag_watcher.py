import os
import time
from fastapi import FastAPI
from mlflow.tracking import MlflowClient

app = FastAPI(title="MLflow Tag Watcher")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "")
WATCH_ALIASES = os.getenv("WATCH_ALIASES", "dev,test,Production")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def _alias_status():
    aliases = [a.strip() for a in WATCH_ALIASES.split(",") if a.strip()]
    status = {}
    for alias in aliases:
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, alias)
            status[alias] = {
                "version": mv.version,
                "run_id": mv.run_id,
                "stage": mv.current_stage,
            }
        except Exception as exc:
            status[alias] = {"error": str(exc)}
    return status


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return {
        "model_name": MODEL_NAME,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "aliases": _alias_status(),
    }
