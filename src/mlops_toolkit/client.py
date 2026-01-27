import os
from typing import Dict, List


def get_client(tracking_uri: str | None = None):
    try:
        from mlflow.tracking import MlflowClient  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mlflow is required. Install with 'pip install mlops-toolkit'.") from exc

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    return MlflowClient(tracking_uri=uri)


def set_alias(model_name: str, version: str, alias: str, tracking_uri: str | None = None) -> None:
    client = get_client(tracking_uri)
    client.set_registered_model_alias(model_name, alias, version)


def get_alias_status(model_name: str, aliases: List[str], tracking_uri: str | None = None) -> Dict:
    client = get_client(tracking_uri)
    status = {}
    for alias in aliases:
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            status[alias] = {"version": mv.version, "run_id": mv.run_id, "stage": mv.current_stage}
        except Exception as exc:
            status[alias] = {"error": str(exc)}
    return status


def log_artifact(run_id: str, path: str, artifact_path: str | None = None, tracking_uri: str | None = None) -> None:
    client = get_client(tracking_uri)
    client.log_artifact(run_id, path, artifact_path=artifact_path)


def download_artifact(run_id: str, artifact_path: str, dst: str, tracking_uri: str | None = None) -> str:
    client = get_client(tracking_uri)
    return client.download_artifacts(run_id, artifact_path, dst_path=dst)
