from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.config import get_settings
from db.base import Base

settings = get_settings()

engine = create_engine(settings.app_db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
