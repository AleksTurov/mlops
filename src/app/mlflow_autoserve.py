import os
import re
import time
from datetime import datetime, timezone
from typing import Iterable

import docker
import mlflow
from mlflow import models as mlflow_models
from mlflow.tracking import MlflowClient

from core.config import get_settings
from core.logger import logger


def _sanitize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9_.-]", "-", value)
    return value[:63]


def _iter_models_with_alias(client: MlflowClient, alias: str) -> Iterable[tuple[str, str]]:
    for model in client.search_registered_models():
        name = model.name
        try:
            version = client.get_model_version_by_alias(name, alias).version
        except Exception:
            continue
        yield name, str(version)


def _image_exists(docker_client: docker.DockerClient, image_name: str) -> bool:
    try:
        docker_client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def _build_model_image(model_uri: str, image_name: str, env_manager: str) -> None:
    mlflow_models.build_docker(
        model_uri=model_uri,
        name=image_name,
        env_manager=env_manager,
    )


def _ensure_container(
    docker_client: docker.DockerClient,
    model_name: str,
    alias: str,
    version: str,
    image: str,
    network: str,
    port: int,
    env: dict,
    serve_mode: str,
    env_manager: str,
) -> None:
    container_name = _sanitize_name(f"mlflow-serve-{model_name}-{alias}")
    container_port = 8080 if serve_mode == "docker-image" else port
    labels = {
        "mlflow_serve": "true",
        "mlflow_model": model_name,
        "mlflow_alias": alias,
        "mlflow_port": str(container_port),
        "mlflow_version": version,
    }

    try:
        container = docker_client.containers.get(container_name)
        current_version = container.labels.get("mlflow_version")
        if current_version != version:
            container.remove(force=True)
            raise docker.errors.NotFound("version changed")
        if container.status != "running":
            container.start()
        logger.info("MLflow serve running: %s (%s@%s v%s)", container_name, model_name, alias, version)
        return
    except docker.errors.NotFound:
        pass

    if serve_mode == "docker-image":
        model_uri = f"models:/{model_name}/{version}"
        image = _sanitize_name(f"mlflow-model-{model_name}-v{version}")
        if not _image_exists(docker_client, image):
            logger.info("Building model image %s from %s", image, model_uri)
            _build_model_image(model_uri=model_uri, image_name=image, env_manager=env_manager)
        command = [
            "mlflow",
            "models",
            "serve",
            "-m",
            "/opt/ml/model",
            "-h",
            "0.0.0.0",
            "-p",
            str(container_port),
            "--env-manager",
            "local",
        ]
        labels["mlflow_image"] = image
    else:
        command = [
            "mlflow",
            "models",
            "serve",
            "-m",
            f"models:/{model_name}@{alias}",
            "-h",
            "0.0.0.0",
            "-p",
            str(container_port),
            "--env-manager",
            env_manager,
        ]

    docker_client.containers.run(
        image=image,
        name=container_name,
        command=command,
        detach=True,
        network=network,
        environment=env,
        labels=labels,
        restart_policy={"Name": "always"},
    )
    logger.info("MLflow serve started: %s (%s@%s v%s)", container_name, model_name, alias, version)


def main() -> None:
    settings = get_settings()

    aliases = [a.strip() for a in os.getenv("MLFLOW_SERVE_ALIASES", "Production").split(",") if a.strip()]
    image = os.getenv("MLFLOW_SERVE_IMAGE", "mlops-mlflow")
    network = os.getenv("MLFLOW_SERVE_NETWORK", "mlops_default")
    port = int(os.getenv("MLFLOW_SERVE_PORT", "5000"))
    poll_seconds = int(os.getenv("MLFLOW_SERVE_POLL_SECONDS", "30"))
    serve_mode = os.getenv("MLFLOW_SERVE_MODE", "docker-image").strip().lower()
    env_manager = os.getenv("MLFLOW_SERVE_ENV_MANAGER", "virtualenv").strip().lower()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    docker_client = docker.from_env()

    env = {
        "MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri,
        "MLFLOW_S3_ENDPOINT_URL": settings.mlflow_s3_endpoint_url,
        "AWS_ACCESS_KEY_ID": settings.aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": settings.aws_secret_access_key,
    }

    logger.info("MLflow autoserve started. Aliases=%s", aliases)
    while True:
        for alias in aliases:
            for model_name, version in _iter_models_with_alias(client, alias):
                _ensure_container(
                    docker_client=docker_client,
                    model_name=model_name,
                    alias=alias,
                    version=version,
                    image=image,
                    network=network,
                    port=port,
                    env=env,
                    serve_mode=serve_mode,
                    env_manager=env_manager,
                )
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
