import os
import uuid
from typing import List

import mlflow
import mlflow.sklearn
import pandas as pd

from core.config import get_settings
from core.logger import logger
from db.crud import get_latest_dataset, insert_predictions, load_dataset_rows
from db.session import SessionLocal, init_db


def _set_mlflow_env(settings) -> None:
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)


def _load_latest_features() -> tuple[pd.DataFrame, str, str, str]:
    init_db()
    with SessionLocal() as db:
        dataset = get_latest_dataset(db)

    if dataset is None:
        raise RuntimeError("No data available for inference.")

    with SessionLocal() as db:
        rows = load_dataset_rows(db, dataset.id)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Dataset has no rows.")
    return df, dataset.name, dataset.task_type, dataset.target_column


def _build_prediction_rows(
    features: pd.DataFrame,
    predictions: List,
    probabilities: List | None,
    model_name: str,
    model_version: str,
    task_type: str,
    model_alias: str | None = None,
    run_group_id: str | None = None,
    model_role: str | None = None,
) -> List[dict]:
    rows = []
    rows_data = features.to_dict(orient="records")
    if task_type == "classification":
        probabilities_iter = probabilities if probabilities is not None else [None] * len(rows_data)
        for row, pred, proba in zip(rows_data, predictions, probabilities_iter):
            proba_values = [float(x) for x in proba] if proba is not None else []
            rows.append(
                {
                    "model_name": model_name,
                    "model_version": str(model_version),
                    "model_alias": model_alias,
                    "run_group_id": run_group_id,
                    "model_role": model_role,
                    "predicted_class": int(pred),
                    "probabilities": {"values": proba_values},
                    "probability_0": float(proba_values[0]) if len(proba_values) > 0 else None,
                    "probability_1": float(proba_values[1]) if len(proba_values) > 1 else None,
                    "probability_2": float(proba_values[2]) if len(proba_values) > 2 else None,
                    "probability_max": float(max(proba_values)) if len(proba_values) > 0 else None,
                    "input_features": row,
                    "sepal_length": row.get("sepal_length"),
                    "sepal_width": row.get("sepal_width"),
                    "petal_length": row.get("petal_length"),
                    "petal_width": row.get("petal_width"),
                }
            )
    else:
        for row, pred in zip(rows_data, predictions):
            rows.append(
                {
                    "model_name": model_name,
                    "model_version": str(model_version),
                    "model_alias": model_alias,
                    "run_group_id": run_group_id,
                    "model_role": model_role,
                    "predicted_value": float(pred),
                    "input_features": row,
                    "sepal_length": row.get("sepal_length"),
                    "sepal_width": row.get("sepal_width"),
                    "petal_length": row.get("petal_length"),
                    "petal_width": row.get("petal_width"),
                }
            )
    return rows


def run_inference() -> int:
    """Load the production model and store predictions in the database."""
    settings = get_settings()
    _set_mlflow_env(settings)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    dataset_df, dataset_name, task_type, target_column = _load_latest_features()
    model_name = f"{settings.mlflow_model_name}_{dataset_name}"
    try:
        model_version = client.get_model_version_by_alias(
            model_name, settings.mlflow_model_alias
        ).version
    except mlflow.exceptions.RestException as exc:
        logger.warning(
            "Inference skipped: alias %s not found (%s)",
            settings.mlflow_model_alias,
            exc,
        )
        return 0

    model_uri = f"models:/{model_name}@{settings.mlflow_model_alias}"
    model = mlflow.sklearn.load_model(model_uri)

    features = dataset_df.drop(columns=[target_column]) if target_column in dataset_df.columns else dataset_df
    preds = model.predict(features)
    probas = model.predict_proba(features) if task_type == "classification" else None

    rows = _build_prediction_rows(
        features,
        preds,
        probas,
        model_name,
        model_version,
        task_type,
        model_alias=settings.mlflow_model_alias,
        model_role="production",
    )

    with SessionLocal() as db:
        insert_predictions(db, rows)

    logger.info("Stored %s predictions.", len(rows))
    return len(rows)


def run_shadow_inference(test_alias: str = "test") -> int:
    """Run inference with production and test aliases and store paired predictions."""
    settings = get_settings()
    _set_mlflow_env(settings)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    dataset_df, dataset_name, task_type, target_column = _load_latest_features()
    model_name = f"{settings.mlflow_model_name}_{dataset_name}"

    try:
        prod_version = client.get_model_version_by_alias(model_name, settings.mlflow_model_alias).version
    except mlflow.exceptions.RestException as exc:
        logger.warning(
            "Shadow inference skipped: alias %s not found (%s)",
            settings.mlflow_model_alias,
            exc,
        )
        return 0
    try:
        test_version = client.get_model_version_by_alias(model_name, test_alias).version
    except mlflow.exceptions.RestException as exc:
        logger.warning("Shadow inference skipped: alias %s not found (%s)", test_alias, exc)
        return 0

    prod_uri = f"models:/{model_name}@{settings.mlflow_model_alias}"
    test_uri = f"models:/{model_name}@{test_alias}"

    prod_model = mlflow.sklearn.load_model(prod_uri)
    test_model = mlflow.sklearn.load_model(test_uri)

    features = dataset_df.drop(columns=[target_column]) if target_column in dataset_df.columns else dataset_df

    prod_preds = prod_model.predict(features)
    prod_probas = prod_model.predict_proba(features) if task_type == "classification" else None

    test_preds = test_model.predict(features)
    test_probas = test_model.predict_proba(features) if task_type == "classification" else None

    run_group_id = str(uuid.uuid4())

    prod_rows = _build_prediction_rows(
        features,
        prod_preds,
        prod_probas,
        model_name,
        prod_version,
        task_type,
        model_alias=settings.mlflow_model_alias,
        run_group_id=run_group_id,
        model_role="production",
    )

    test_rows = _build_prediction_rows(
        features,
        test_preds,
        test_probas,
        model_name,
        test_version,
        task_type,
        model_alias=test_alias,
        run_group_id=run_group_id,
        model_role="test",
    )

    with SessionLocal() as db:
        insert_predictions(db, prod_rows + test_rows)

    logger.info("Stored %s shadow predictions (group %s).", len(prod_rows) + len(test_rows), run_group_id)
    return len(prod_rows) + len(test_rows)
