import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable

import docker
import mlflow
from mlflow import models as mlflow_models
from mlflow.tracking import MlflowClient

from core.config import get_settings
from core.logger import logger


PROCESS_STARTED_AT = datetime.now(timezone.utc)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return

        payload = {
            "status": "ok",
            "service": "mlflow-autoserve",
            "started_at": PROCESS_STARTED_AT.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        body = json.dumps(payload).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def _start_health_endpoint() -> None:
    host = os.getenv("MLFLOW_AUTOSERVE_HEALTH_HOST", "0.0.0.0")
    port = int(os.getenv("MLFLOW_AUTOSERVE_HEALTH_PORT", "5010"))
    server = HTTPServer((host, port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, name="autoserve-health", daemon=True)
    thread.start()
    logger.info("Autoserve health endpoint started: http://%s:%s/health", host, port)


def _sanitize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9_.-]", "-", value)
    return value[:63]


def _env_alias_key(alias: str) -> str:
    return re.sub(r"[^A-Z0-9]", "_", alias.upper())


def _parse_alias_projects(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not raw:
        return result
    for part in raw.split(","):
        token = part.strip()
        if not token or "=" not in token:
            continue
        alias, project = token.split("=", 1)
        alias = alias.strip()
        project = project.strip()
        if alias and project:
            result[alias.lower()] = project
    return result


def _project_for_alias(alias: str, alias_projects: dict[str, str]) -> str:
    env_key = f"MLFLOW_SERVE_PROJECT_{_env_alias_key(alias)}"
    env_value = os.getenv(env_key, "").strip()
    if env_value:
        return env_value
    return alias_projects.get(alias.lower(), f"models_{_sanitize_name(alias)}")


def _iter_models_with_alias(client: MlflowClient, alias: str) -> Iterable[tuple[str, str]]:
    for model in client.search_registered_models():
        name = model.name
        try:
            version = client.get_model_version_by_alias(name, alias).version
        except Exception:
            continue
        yield name, str(version)


def _get_source_experiment_name(client: MlflowClient, model_name: str, version: str) -> str:
    try:
        model_version = client.get_model_version(model_name, version)
        run_id = getattr(model_version, "run_id", None)
        if not run_id:
            return ""
        run = client.get_run(run_id)
        experiment = client.get_experiment(run.info.experiment_id)
        return experiment.name if experiment else ""
    except Exception:
        return ""


def _resolve_trace_experiment(
    client: MlflowClient,
    model_name: str,
    alias: str,
    version: str,
    default_experiment: str,
) -> str:
    source_experiment = _get_source_experiment_name(client, model_name, version)
    if source_experiment:
        return source_experiment

    if default_experiment:
        logger.warning(
            "Trace experiment fallback used for %s@%s v%s",
            model_name,
            alias,
            version,
        )
    return default_experiment


def _image_exists(docker_client: docker.DockerClient, image_name: str) -> bool:
    try:
        docker_client.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False


def _build_base_image() -> str:
    return os.getenv("MLFLOW_SERVE_BUILD_BASE_IMAGE", "mlops-mlflow-autoserve")


def _build_image_name(docker_client: docker.DockerClient, model_name: str, version: str) -> str:
    base_image = _build_base_image()
    try:
        variant_source = docker_client.images.get(base_image).id
    except docker.errors.ImageNotFound:
        variant_source = base_image
    variant = hashlib.sha1(variant_source.encode("utf-8")).hexdigest()[:8]
    return _sanitize_name(f"mlflow-model-{model_name}-v{version}-{variant}")


def _build_model_image(model_uri: str, image_name: str, env_manager: str) -> None:
    mlflow_models.build_docker(
        model_uri=model_uri,
        name=image_name,
        env_manager=env_manager,
        base_image=_build_base_image(),
    )


def _build_model_image_with_retries(
    model_uri: str,
    image_name: str,
    env_manager: str,
    retries: int,
    retry_delay_seconds: int,
) -> None:
    total_attempts = max(1, retries)
    for attempt in range(1, total_attempts + 1):
        try:
            _build_model_image(model_uri=model_uri, image_name=image_name, env_manager=env_manager)
            return
        except Exception:
            if attempt >= total_attempts:
                raise
            logger.warning(
                "Model image build failed for %s (attempt %d/%d). Retrying in %ds",
                image_name,
                attempt,
                total_attempts,
                retry_delay_seconds,
            )
            time.sleep(retry_delay_seconds)


def _gpu_enabled() -> bool:
    return os.getenv("MLFLOW_SERVE_ENABLE_GPU", "false").strip().lower() in {"1", "true", "yes"}


def _ensure_container(
    docker_client: docker.DockerClient,
    model_name: str,
    alias: str,
    version: str,
    project: str,
    image: str,
    network: str,
    port: int,
    env: dict,
    serve_mode: str,
    env_manager: str,
    build_retries: int,
    build_retry_delay_seconds: int,
    traces_experiment_name: str,
) -> None:
    container_name = _sanitize_name(f"mlflow-serve-{model_name}-{alias}")
    container_port = 8080 if serve_mode == "docker-image" else port
    desired_image = image
    labels = {
        "mlflow_serve": "true",
        "mlflow_model": model_name,
        "mlflow_alias": alias,
        "mlflow_port": str(container_port),
        "mlflow_version": version,
        "mlflow_serve_mode": serve_mode,
        "mlflow_gpu_enabled": str(_gpu_enabled()).lower(),
        "mlflow_models_workers": env.get("MLFLOW_MODELS_WORKERS", "1"),
        "mlflow_launch_mode": "direct-cli" if serve_mode == "docker-image" else "image-default",
        "com.docker.compose.project": project,
        "com.docker.compose.service": "mlflow-serve",
    }

    if serve_mode == "docker-image":
        desired_image = _build_image_name(docker_client, model_name, version)
        labels["mlflow_image"] = desired_image

    try:
        container = docker_client.containers.get(container_name)
        current_version = container.labels.get("mlflow_version")
        current_project = container.labels.get("com.docker.compose.project")
        current_mode = container.labels.get("mlflow_serve_mode")
        current_image = container.labels.get("mlflow_image")
        current_gpu_enabled = container.labels.get("mlflow_gpu_enabled")
        current_models_workers = container.labels.get("mlflow_models_workers")
        current_launch_mode = container.labels.get("mlflow_launch_mode")
        if (
            current_version != version
            or current_project != project
            or current_mode != serve_mode
            or current_image != labels.get("mlflow_image")
            or current_gpu_enabled != labels.get("mlflow_gpu_enabled")
            or current_models_workers != labels.get("mlflow_models_workers")
            or current_launch_mode != labels.get("mlflow_launch_mode")
        ):
            container.remove(force=True)
            raise docker.errors.NotFound(
                "version, project, serve mode, image, gpu mode, workers or launch mode changed"
            )
        if container.status != "running":
            container.remove(force=True)
            raise docker.errors.NotFound("container not running")
        logger.info("MLflow serve running: %s (%s@%s v%s)", container_name, model_name, alias, version)
        return
    except docker.errors.NotFound:
        pass

    entrypoint = None
    if serve_mode == "docker-image":
        model_uri = f"models:/{model_name}/{version}"
        image = desired_image
        if not _image_exists(docker_client, image):
            logger.info("Building model image %s from %s", image, model_uri)
            _build_model_image_with_retries(
                model_uri=model_uri,
                image_name=image,
                env_manager=env_manager,
                retries=build_retries,
                retry_delay_seconds=build_retry_delay_seconds,
            )
        command = [
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
        entrypoint = ["mlflow"]
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

    container_env = dict(env)
    container_env["MLFLOW_MODEL_ALIAS"] = alias
    container_env["MLFLOW_MODEL_NAME"] = model_name
    container_env["MLFLOW_MODEL_VERSION"] = str(version)
    if traces_experiment_name:
        container_env["MLFLOW_EXPERIMENT_NAME"] = traces_experiment_name

    device_requests = None
    if _gpu_enabled():
        container_env["NVIDIA_VISIBLE_DEVICES"] = "all"
        container_env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

    docker_client.containers.run(
        image=image,
        name=container_name,
        command=command,
        detach=True,
        network=network,
        environment=container_env,
        labels=labels,
        restart_policy={"Name": "always"},
        device_requests=device_requests,
        entrypoint=entrypoint,
    )
    logger.info("MLflow serve started: %s (%s@%s v%s)", container_name, model_name, alias, version)


def main() -> None:
    settings = get_settings()
    _start_health_endpoint()

    aliases = [a.strip() for a in os.getenv("MLFLOW_SERVE_ALIASES", "champion").split(",") if a.strip()]
    alias_projects = _parse_alias_projects(
        os.getenv(
            "MLFLOW_SERVE_ALIAS_PROJECTS",
            "champion=models_champion,challenger=models_challenger",
        )
    )
    image = os.getenv("MLFLOW_SERVE_IMAGE", "mlops-mlflow")
    network = os.getenv("MLFLOW_SERVE_NETWORK", "mlops_default")
    port = int(os.getenv("MLFLOW_SERVE_PORT", "5000"))
    poll_seconds = int(os.getenv("MLFLOW_SERVE_POLL_SECONDS", "30"))
    serve_mode = os.getenv("MLFLOW_SERVE_MODE", "docker-image").strip().lower()
    env_manager = os.getenv("MLFLOW_SERVE_ENV_MANAGER", "virtualenv").strip().lower()
    build_retries = int(os.getenv("MLFLOW_SERVE_BUILD_RETRIES", "5"))
    build_retry_delay_seconds = int(os.getenv("MLFLOW_SERVE_BUILD_RETRY_DELAY_SECONDS", "20"))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    docker_client = docker.from_env()

    env = {
        "MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri,
        "MLFLOW_S3_ENDPOINT_URL": settings.mlflow_s3_endpoint_url,
        "AWS_ACCESS_KEY_ID": settings.aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": settings.aws_secret_access_key,
        "DISABLE_NGINX": "true",
        "MLFLOW_MODELS_WORKERS": os.getenv("MLFLOW_MODELS_WORKERS", "1"),
    }
    traces_experiment_default = os.getenv("MLFLOW_TRACES_EXPERIMENT_NAME", "").strip()

    logger.info("MLflow autoserve started. Aliases=%s", aliases)
    while True:
        for alias in aliases:
            project = _project_for_alias(alias, alias_projects)
            for model_name, version in _iter_models_with_alias(client, alias):
                try:
                    trace_experiment_name = _resolve_trace_experiment(
                        client=client,
                        model_name=model_name,
                        alias=alias,
                        version=version,
                        default_experiment=traces_experiment_default,
                    )
                    _ensure_container(
                        docker_client=docker_client,
                        model_name=model_name,
                        alias=alias,
                        version=version,
                        project=project,
                        image=image,
                        network=network,
                        port=port,
                        env=env,
                        serve_mode=serve_mode,
                        env_manager=env_manager,
                        build_retries=build_retries,
                        build_retry_delay_seconds=build_retry_delay_seconds,
                        traces_experiment_name=trace_experiment_name,
                    )
                except Exception:
                    logger.exception("Autoserve reconcile failed for %s@%s", model_name, alias)
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
