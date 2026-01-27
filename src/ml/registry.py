from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

import mlflow

from core.config import get_settings
from core.logger import logger
from db.crud import get_latest_dataset
from db.session import SessionLocal, init_db


@dataclass
class ModelRegistryState:
    alias: str
    version: Optional[str]
    last_checked: datetime


class ModelRegistryCache:
    """In-memory cache for the current MLflow model alias."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._state: Optional[ModelRegistryState] = None

    def refresh(self) -> ModelRegistryState:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()

        init_db()
        with SessionLocal() as db:
            dataset = get_latest_dataset(db)
        model_name = (
            f"{settings.mlflow_model_name}_{dataset.name}" if dataset else settings.mlflow_model_name
        )

        try:
            version_info = client.get_model_version_by_alias(
                model_name, settings.mlflow_model_alias
            )
            version = version_info.version
        except Exception as exc:
            logger.warning("Unable to fetch model alias: %s", exc)
            version = None

        with self._lock:
            self._state = ModelRegistryState(
                alias=settings.mlflow_model_alias,
                version=str(version) if version is not None else None,
                last_checked=datetime.now(timezone.utc),
            )
            return self._state

    def get(self) -> ModelRegistryState:
        with self._lock:
            if self._state is None:
                return self.refresh()
            return self._state


model_registry_cache = ModelRegistryCache()
