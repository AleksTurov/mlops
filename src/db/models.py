from sqlalchemy import Column, Date, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from db.base import Base


class IrisData(Base):
    """Raw Iris dataset stored in the application database."""

    __tablename__ = "iris_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    target = Column(Integer, nullable=False)
    batch_date = Column(Date, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    """Model predictions stored in the application database."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(64), nullable=False)
    model_alias = Column(String(32), nullable=True)
    run_group_id = Column(String(64), nullable=True)
    model_role = Column(String(32), nullable=True)
    sepal_length = Column(Float, nullable=True)
    sepal_width = Column(Float, nullable=True)
    petal_length = Column(Float, nullable=True)
    petal_width = Column(Float, nullable=True)
    input_features = Column(JSONB, nullable=True)
    predicted_class = Column(Integer, nullable=True)
    predicted_value = Column(Float, nullable=True)
    probabilities = Column(JSONB, nullable=True)
    probability_0 = Column(Float, nullable=True)
    probability_1 = Column(Float, nullable=True)
    probability_2 = Column(Float, nullable=True)
    probability_max = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Dataset(Base):
    """Dataset metadata stored in the application database."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    task_type = Column(String(32), nullable=False)
    target_column = Column(String(128), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DatasetRow(Base):
    """Dataset rows stored as JSON for flexible schemas."""

    __tablename__ = "dataset_rows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(ForeignKey("datasets.id"), nullable=False)
    row = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ABWeight(Base):
    """Persistent storage for A/B routing weight (percentage of traffic to prod)."""

    __tablename__ = "ab_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prod_percent = Column(Integer, nullable=False, default=80)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
