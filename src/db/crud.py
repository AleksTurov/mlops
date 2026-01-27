from datetime import date
from typing import Iterable, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Dataset, DatasetRow, IrisData, Prediction


def create_dataset(db: Session, name: str, task_type: str, target_column: str) -> Dataset:
    """Create a dataset metadata record."""
    dataset = Dataset(name=name, task_type=task_type, target_column=target_column)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def insert_dataset_rows(db: Session, dataset_id: int, rows: Iterable[dict]) -> None:
    """Insert dataset rows as JSONB documents."""
    payload = [{"dataset_id": dataset_id, "row": row} for row in rows]
    db.bulk_insert_mappings(DatasetRow, payload)
    db.commit()


def get_latest_dataset(db: Session) -> Optional[Dataset]:
    """Return the latest dataset metadata record."""
    stmt = select(Dataset).order_by(Dataset.created_at.desc()).limit(1)
    return db.execute(stmt).scalars().first()


def get_dataset(db: Session, dataset_id: int) -> Optional[Dataset]:
    """Return a dataset by ID."""
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    return db.execute(stmt).scalars().first()


def load_dataset_rows(db: Session, dataset_id: int) -> List[dict]:
    """Load dataset rows by dataset ID."""
    stmt = select(DatasetRow.row).where(DatasetRow.dataset_id == dataset_id)
    return list(db.execute(stmt).scalars().all())


def get_latest_batch_date(db: Session) -> Optional[date]:
    """Return the latest batch date stored in the Iris table."""
    stmt = select(func.max(IrisData.batch_date))
    return db.execute(stmt).scalar_one_or_none()


def insert_iris_data(db: Session, rows: Iterable[dict]) -> None:
    """Insert Iris rows into the database."""
    db.bulk_insert_mappings(IrisData, list(rows))
    db.commit()


def insert_predictions(db: Session, rows: Iterable[dict]) -> None:
    """Insert prediction rows into the database."""
    db.bulk_insert_mappings(Prediction, list(rows))
    db.commit()


def list_predictions(db: Session, limit: int, offset: int) -> List[Prediction]:
    """Return a paginated list of predictions."""
    stmt = select(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset)
    return list(db.execute(stmt).scalars().all())


def get_prediction(db: Session, prediction_id: int) -> Optional[Prediction]:
    """Return a single prediction by ID."""
    stmt = select(Prediction).where(Prediction.id == prediction_id)
    return db.execute(stmt).scalars().first()
