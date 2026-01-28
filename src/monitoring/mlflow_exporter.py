import os
import time
from typing import Dict, List, Set, Tuple

from mlflow.tracking import MlflowClient
from prometheus_client import Gauge, start_http_server

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPORTER_PORT = int(os.getenv("MLFLOW_EXPORTER_PORT", "9101"))
POLL_SECONDS = int(os.getenv("MLFLOW_EXPORTER_POLL_SECONDS", "30"))
ALIASES = [a.strip() for a in os.getenv("MLFLOW_EXPORTER_ALIASES", "Production").split(",") if a.strip()]

models_total = Gauge("mlflow_registered_models_total", "Total registered models")
model_versions_total = Gauge(
    "mlflow_model_versions_total",
    "Total model versions per model",
    ["model_name"],
)
model_version_created = Gauge(
    "mlflow_model_version_created_timestamp",
    "Model version creation timestamp (ms since epoch)",
    ["model_name", "version"],
)
model_alias_version = Gauge(
    "mlflow_model_alias_version",
    "Model version for alias",
    ["model_name", "alias", "version"],
)
last_success = Gauge("mlflow_exporter_last_success_timestamp", "Last successful scrape timestamp")


def _safe_remove(gauge: Gauge, labels: Set[Tuple[str, ...]], keep: Set[Tuple[str, ...]]) -> None:
    for label in labels - keep:
        try:
            gauge.remove(*label)
        except KeyError:
            pass


def _collect(client: MlflowClient) -> None:
    registered = client.search_registered_models()
    models_total.set(len(registered))

    current_versions: Set[Tuple[str, ...]] = set()
    current_created: Set[Tuple[str, ...]] = set()
    current_aliases: Set[Tuple[str, ...]] = set()

    for model in registered:
        name = model.name
        versions = client.search_model_versions(f"name='{name}'")
        model_versions_total.labels(model_name=name).set(len(versions))
        current_versions.add((name,))

        for version in versions:
            v = str(version.version)
            model_version_created.labels(model_name=name, version=v).set(version.creation_timestamp)
            current_created.add((name, v))

        for alias in ALIASES:
            try:
                alias_version = client.get_model_version_by_alias(name, alias)
                v = str(alias_version.version)
                model_alias_version.labels(model_name=name, alias=alias, version=v).set(1)
                current_aliases.add((name, alias, v))
            except Exception:
                continue

    _safe_remove(model_versions_total, set(model_versions_total._metrics.keys()), current_versions)
    _safe_remove(model_version_created, set(model_version_created._metrics.keys()), current_created)
    _safe_remove(model_alias_version, set(model_alias_version._metrics.keys()), current_aliases)


def main() -> None:
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    start_http_server(EXPORTER_PORT)

    while True:
        try:
            _collect(client)
            last_success.set(time.time())
        except Exception:
            pass
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
