import numpy as np
import torch
from torch.utils.data import Dataset
    
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float, Date, Boolean, BigInteger
from clickhouse_sqlalchemy.types import Array
from clickhouse_sqlalchemy import engines, types
from pydantic import BaseModel, Field, conint
from typing import Optional
from datetime import date, datetime
import uuid

Base = declarative_base()


class LSTMCreditScoringDataset(Dataset):
    """
    PyTorch Dataset для LSTM модели скорингования кредитов.
    Хранит данные как списки тензоров.
    """
    def __init__(self, X_num_list, X_cat_list, y_targets, lengths, metadata=None):
        self.Xn_list = X_num_list
        self.Xc_list = X_cat_list
        self.y = torch.from_numpy(y_targets.astype(np.float32)).float()
        self.lengths = lengths
        self.metadata = metadata
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            'X_num': self.Xn_list[idx],
            'X_cat': self.Xc_list[idx],
            'y': self.y[idx],
            'len': self.lengths[idx],
            'idx': idx
        }


class InsertPredictionsScoring(Base):
    ''' Таблица для хранения предсказаний скоринговой модели.'''
    __tablename__ = 'credit_scoring_predictions'
    __table_args__ = (
        engines.MergeTree(order_by=("subs_id", "ctn", "insert_datetime")),
        {"schema": "data_science"},

    )
    subs_id = Column(BigInteger, primary_key=True, nullable=False, comment='Идентификатор абонента')  
    ctn = Column(BigInteger, primary_key=True, nullable=False, comment='Номер телефона клиента')  
    insert_datetime = Column(DateTime, primary_key=True, nullable=False, comment='Дата рассчета предсказания')
    probability = Column(Float, nullable=False, comment='Вероятность дефолта')
    observation_period_end_date = Column(Date, nullable=False, comment='Дата конца наблюдаемого периода')
    load_dt = Column(DateTime, nullable=False, comment='Дата и время загрузки записи в таблицу')


class APIRequestLog(Base):
    ''' Таблица логов API запросов. '''
    __tablename__ = 'api_requests_log'
    __table_args__ = (
        engines.MergeTree(order_by=("request_id", "request_datetime")),
        {"schema": "data_science"},
    )
    request_id = Column(String, primary_key=True, nullable=False, comment='Уникальный идентификатор запроса')
    request_datetime = Column(DateTime, nullable=False, comment='Дата и время запроса')
    phone = Column(types.Nullable(types.Int64), nullable=False, comment='Телефон клиента')
    settlement_date = Column(types.Nullable(types.Date), comment='Дата для предсказания')
    status = Column(String, nullable=False, comment='Статус ответа')
    description = Column(String, default=None, nullable=True, comment='Описание ошибки или статуса')
    probability = Column(types.Nullable(types.Float64), comment='Вероятность дефолта')
    subs_id = Column(types.Nullable(types.Int64), comment='ID абонента')
    insert_datetime = Column(types.Nullable(types.DateTime), comment='Дата и время вставки записи')
    observation_period_end_date = Column(types.Nullable(types.Date), comment='Дата окончания периода наблюдения')
    time_taken_ms = Column(BigInteger, nullable=False, comment='Время обработки запроса в миллисекундах')


# --- Pydantic схемы ---
class PredictionRequest(BaseModel):
    ''' Схема запроса предсказания. '''
    phone: conint(ge=0) = Field(..., description="Телефон клиента (BigInteger)")
    settlement_date: Optional[date] = Field(default_factory=date.today, description="Дата для выборки (по умолчанию сегодня)")

class PredictionResponse(PredictionRequest):
    ''' Схема ответа предсказания. '''
    probability: Optional[float] = Field(None, description="Вероятность дефолта или None")
    status: str = Field(default='error', description="found|not_found|error|success")
    request_id: str = Field(default=str(uuid.uuid4()), description="Уникальный идентификатор запроса")


# Создадим таблицы (если еще нет)
from src.database import clickhouse_engine
try:
    Base.metadata.create_all(clickhouse_engine)
except Exception as e:
    print(f"Не удалось создать таблицу логов: {e}")
