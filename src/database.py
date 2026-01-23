from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from sqlalchemy.pool import QueuePool

from src.config import config
from src.logger import logger
import urllib3

# Отключаем SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- ENGINES ---
postgres_engine = None
clickhouse_engine = None
ipdr_engine = None

# PostgreSQL
try:
    postgres_engine = create_engine(config.database.postgres_url)
    logger.info("✅ PostgreSQL engine создан")
except Exception as e:
    logger.error(f"❌ PostgreSQL: {e}")

# ClickHouse с SSL параметрами
try:
    clickhouse_engine = create_engine(
        config.database.clickhouse_url,
        connect_args=config.database.ssl_args,
        poolclass=QueuePool,
        pool_size=10,        # базовый размер пула (увеличить при необходимости)
        max_overflow=20,     # сколько сверх pool_size можно создать временно
        pool_timeout=120,     # ждать свободное соединение до таймаута (сек)
        pool_pre_ping=True   # проверять живость соединения перед использованием
    )
    logger.info("✅ ClickHouse engine создан")
except Exception as e:
    logger.error(f"❌ ClickHouse: {e}")

# IPDR ClickHouse
try:
    ipdr_engine = create_engine(
        config.database.clickhouse_ipdr_url,
        connect_args=config.database.ssl_args
    )
    logger.info("✅ IPDR ClickHouse engine создан")
except Exception as e:
    logger.error(f"❌ IPDR ClickHouse: {e}")

# --- SESSIONS ---
postgres_session_factory = sessionmaker(bind=postgres_engine) if postgres_engine else None
clickhouse_session_factory = sessionmaker(bind=clickhouse_engine) if clickhouse_engine else None
ipdr_session_factory = sessionmaker(bind=ipdr_engine) if ipdr_engine else None

# Base для моделей
Base = declarative_base()

# --- SIMPLE TEST ---
def test_connections():
    """Простой тест подключений."""
    results = {}
    
    for name, engine in [('postgres', postgres_engine), 
                        ('clickhouse', clickhouse_engine), 
                        ('ipdr', ipdr_engine)]:
        if engine:
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                results[name] = '✅'
            except Exception as e:
                results[name] = f'❌ {str(e)[:50]}...'
        else:
            results[name] = '❌ не создан'
    
    return results

# --- TEST SCRIPT ---
if __name__ == "__main__":
    logger.info("Тест подключений:")
    results = test_connections()
    for db, status in results.items():
        logger.info(f"{db}: {status}")



