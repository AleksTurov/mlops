import os
from dataclasses import dataclass
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_db_url: str = os.getenv(
        "APP_DB_URL",
        "postgresql+psycopg2://app_user:app_password@app-db:5432/app_db",
    )
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification")
    mlflow_model_name: str = os.getenv("MLFLOW_MODEL_NAME", "iris_classifier")
    mlflow_model_alias: str = os.getenv("MLFLOW_MODEL_ALIAS", "Production")
    mlflow_s3_endpoint_url: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    s3_artifact_bucket: str = os.getenv("S3_ARTIFACT_BUCKET", "mlflow")
    calibration_method: str = os.getenv("CALIBRATION_METHOD", "sigmoid")
    model_poll_seconds: int = int(os.getenv("MODEL_POLL_SECONDS", "300"))
    model_server_poll_seconds: int = int(os.getenv("MODEL_SERVER_POLL_SECONDS", "300"))
    model_alias: str = os.getenv("MODEL_ALIAS", "Production")
    model_role: str = os.getenv("MODEL_ROLE", "production")
    model_name_override: str = os.getenv("MODEL_NAME", "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
