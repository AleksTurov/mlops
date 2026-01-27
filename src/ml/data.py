from datetime import date
from typing import Optional

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

from core.logger import logger
from db.crud import create_dataset, insert_dataset_rows
from db.session import SessionLocal, init_db


def _store_dataframe(df: pd.DataFrame, name: str, task_type: str, target_column: str) -> int:
    init_db()
    df = df.where(pd.notnull(df), None)
    rows = df.to_dict(orient="records")
    with SessionLocal() as db:
        dataset = create_dataset(db, name=name, task_type=task_type, target_column=target_column)
        insert_dataset_rows(db, dataset.id, rows)
    logger.info("Dataset %s stored: %s rows", name, len(rows))
    return dataset.id


def load_sklearn_dataset(name: str, task_type: Optional[str] = None) -> int:
    """Load an open dataset from sklearn and store it in the database."""
    name = name.lower()
    if name == "iris":
        dataset = load_iris(as_frame=True)
        default_task = "classification"
    elif name == "wine":
        dataset = load_wine(as_frame=True)
        default_task = "classification"
    elif name in {"breast_cancer", "cancer"}:
        dataset = load_breast_cancer(as_frame=True)
        default_task = "classification"
    elif name == "diabetes":
        dataset = load_diabetes(as_frame=True)
        default_task = "regression"
    else:
        raise ValueError("Unsupported dataset. Use iris, wine, breast_cancer, or diabetes.")

    task_type = task_type or default_task
    df = dataset.frame.copy()
    target_column = dataset.target_names if hasattr(dataset, "target_names") else "target"
    if isinstance(target_column, list):
        target_column = "target"
    df.columns = [*dataset.feature_names, "target"]
    return _store_dataframe(df, name=name, task_type=task_type, target_column="target")


def load_csv_dataset(df: pd.DataFrame, name: str, target_column: str, task_type: str) -> int:
    """Store a user-provided dataframe in the database."""
    if target_column not in df.columns:
        raise ValueError("Target column not found in dataset.")
    return _store_dataframe(df, name=name, task_type=task_type, target_column=target_column)
