import json
import os
import tempfile
from typing import Any, Dict, Tuple, Literal, cast

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR

from core.config import get_settings
from core.logger import logger
from db.crud import get_dataset, get_latest_dataset, load_dataset_rows
from db.session import SessionLocal, init_db


def _set_mlflow_env(settings) -> None:
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)


def _load_dataset(dataset_id: int | None = None) -> Tuple[pd.DataFrame, str, str, str]:
    init_db()
    with SessionLocal() as db:
        dataset = get_dataset(db, dataset_id) if dataset_id else get_latest_dataset(db)
        if dataset is None:
            raise RuntimeError("No dataset found. Load a dataset first.")
        rows = load_dataset_rows(db, dataset.id)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Dataset has no rows.")

    return df, dataset.name, dataset.task_type, dataset.target_column


def _build_models(task_type: str, random_state: int) -> Dict[str, Any]:
    if task_type == "regression":
        return {
            "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=random_state),
            "Ridge": Ridge(random_state=random_state),
            "SVR": SVR(kernel="rbf"),
        }
    return {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=500, multi_class="auto"),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
    }


def _metric_name(task_type: str) -> str:
    return "rmse" if task_type == "regression" else "accuracy"


def _evaluate(task_type: str, y_true, y_pred) -> float:
    if task_type == "regression":
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        return float(rmse)
    return float(accuracy_score(y_true, y_pred))


def train_candidate(dataset_id: int | None = None, random_state: int = 42, test_size: float = 0.2) -> Tuple[str, float, str, str]:
    """Train multiple models and register the best model as a test candidate."""
    settings = get_settings()
    _set_mlflow_env(settings)

    df, dataset_name, task_type, target_column = _load_dataset(dataset_id)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_name = f"{settings.mlflow_experiment_name}_{dataset_name}"
    mlflow.set_experiment(experiment_name)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == "classification" else None
    )

    best_score = None
    best_name = ""
    best_model: Any = None

    models = _build_models(task_type, random_state)
    metric_name = _metric_name(task_type)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            score = _evaluate(task_type, y_valid, preds)

            mlflow.log_param("model_name", name)
            mlflow.log_param("task_type", task_type)
            mlflow.log_metric(metric_name, score)
            mlflow.sklearn.log_model(model, artifact_path="model")

            logger.info("Model %s %s: %.4f", name, metric_name, score)

            if best_score is None:
                best_score = score
                best_name = name
                best_model = model
            else:
                if task_type == "regression" and score < best_score:
                    best_score = score
                    best_name = name
                    best_model = model
                if task_type == "classification" and score > best_score:
                    best_score = score
                    best_name = name
                    best_model = model

    if best_model is None or best_score is None:
        raise RuntimeError("No model was trained successfully.")

    if task_type == "classification":
        method_value = settings.calibration_method if settings.calibration_method in {"sigmoid", "isotonic"} else "sigmoid"
        method = cast(Literal["sigmoid", "isotonic"], method_value)
        calibrated = CalibratedClassifierCV(best_model, method=method, cv="prefit")
        calibrated.fit(X_train, y_train)
        calibrated_preds = calibrated.predict(X_valid)
        calibrated_score = _evaluate(task_type, y_valid, calibrated_preds)
        final_model = calibrated
        final_score = calibrated_score
    else:
        final_model = best_model
        final_score = best_score

    model_name = f"{settings.mlflow_model_name}_{dataset_name}"

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_param("base_model", best_name)
        mlflow.log_param("task_type", task_type)
        mlflow.log_metric(metric_name, final_score)
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("target_column", target_column)

        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = os.path.join(tmpdir, "input_schema.json")
            sample_path = os.path.join(tmpdir, "sample_input.csv")
            metrics_path = os.path.join(tmpdir, "validation_metrics.json")

            input_schema = {
                "dataset": dataset_name,
                "task_type": task_type,
                "target_column": target_column,
                "features": list(X.columns),
            }
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(input_schema, f, ensure_ascii=False, indent=2)

            X_valid.head(50).to_csv(sample_path, index=False)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump({metric_name: final_score}, f, ensure_ascii=False, indent=2)

            mlflow.log_artifact(schema_path, artifact_path="data_contract")
            mlflow.log_artifact(sample_path, artifact_path="data_contract")
            mlflow.log_artifact(metrics_path, artifact_path="metrics")

        model_info = mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=model_name,
        )

    logger.info("Best model: %s (%s %.4f)", best_name, metric_name, final_score)
    return model_name, final_score, str(model_info.registered_model_version), experiment_name


def evaluate_models(dataset_id: int | None = None) -> Dict[str, Any]:
    """Evaluate candidate vs production model on the latest dataset."""
    settings = get_settings()
    _set_mlflow_env(settings)

    df, dataset_name, task_type, target_column = _load_dataset(dataset_id)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    model_name = f"{settings.mlflow_model_name}_{dataset_name}"
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    metric_name = _metric_name(task_type)
    results: Dict[str, Any] = {"metric": metric_name, "task_type": task_type, "model_name": model_name}

    candidate = client.search_model_versions(f"name='{model_name}'")
    candidate_versions = [m for m in candidate if dict(m.tags).get("stage") == "test"]
    if candidate_versions:
        candidate_version = max(candidate_versions, key=lambda m: int(m.version))
        candidate_uri = f"models:/{model_name}/{candidate_version.version}"
        candidate_model = mlflow.sklearn.load_model(candidate_uri)
        candidate_preds = candidate_model.predict(X)
        candidate_score = _evaluate(task_type, y, candidate_preds)
        results["candidate_version"] = candidate_version.version
        results["candidate_score"] = candidate_score
    else:
        results["candidate_version"] = None
        results["candidate_score"] = None

    try:
        prod_version = client.get_model_version_by_alias(model_name, settings.mlflow_model_alias)
        prod_uri = f"models:/{model_name}/{prod_version.version}"
        prod_model = mlflow.sklearn.load_model(prod_uri)
        prod_preds = prod_model.predict(X)
        prod_score = _evaluate(task_type, y, prod_preds)
        results["production_version"] = prod_version.version
        results["production_score"] = prod_score
    except Exception:
        results["production_version"] = None
        results["production_score"] = None

    return results


def promote_if_better(dataset_id: int | None = None) -> dict:
    """Return a recommendation to promote in MLflow UI."""
    metrics = evaluate_models(dataset_id)
    candidate_score = metrics.get("candidate_score")
    production_score = metrics.get("production_score")

    if candidate_score is None:
        return {"recommendation": "no_candidate", **metrics}

    better = False
    if metrics["task_type"] == "regression":
        if production_score is None or candidate_score < production_score:
            better = True
    else:
        if production_score is None or candidate_score > production_score:
            better = True

    return {"recommendation": "promote" if better else "keep", **metrics}


def set_model_alias(model_name: str, version: str, alias: str) -> dict:
    """Set MLflow model alias explicitly via API."""
    settings = get_settings()
    _set_mlflow_env(settings)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)

    return {"model_name": model_name, "alias": alias, "version": version}
