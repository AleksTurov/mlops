"""
Конфигурационный модуль для проекта KPI Models.
Содержит настройки подключений к базам данных, пути к файлам и параметры обработки данных.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote_plus
from dotenv import load_dotenv
from datetime import datetime

# Загрузка переменных окружения
env_path = Path(".env")
loaded = load_dotenv(env_path)
# загрузка переменных окружения для MLflow
load_dotenv(dotenv_path=Path('/data/aturov/mlflow/.env'))
# Базовый путь проекта
BASE_PATH = Path(__file__).parent.parent

#######################
# ENVIRONMENT CONFIG  #
#######################

@dataclass
class EnvironmentConfig:
    """Настройки окружения и путей к данным."""
    
    data_raw_path: Path = Path(os.getenv("DATA_RAW_PATH", str(BASE_PATH / "data" / "raw")))
    data_processed_path: Path = Path(os.getenv("DATA_PROCESSED_PATH", str(BASE_PATH / "data" / "processed")))
    data_final_path: Path = Path(os.getenv("OUTPUT_PATH", str(BASE_PATH / "data" / "final")))
    model_path: Path = Path(os.getenv("MODEL_PATH", str(BASE_PATH / "models"))) # Путь для сохранения моделей
    model_path_raw: Path = Path(os.getenv("MODEL_PATH_RAW", str(BASE_PATH / "models" / "raw")))
    model_path_processed: Path = Path(os.getenv("MODEL_PATH_PROCESSED", str(BASE_PATH / "models" / "processed")))
    model_path_final: Path = Path(os.getenv("MODEL_PATH_FINAL", str(BASE_PATH / "models" / "final")))
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", str(BASE_PATH / "artifacts"))) # Путь для артефактов MLflow
    def get_data_paths(self) -> Dict[str, Path]:
        """Возвращает словарь всех путей к данным."""
        return {
            'raw': self.data_raw_path,
            'processed': self.data_processed_path,
            'final': self.data_final_path
        }
    
    def create_directories(self) -> None:
        """Создает все необходимые директории."""
        for p in self.get_data_paths().values():
            # p уже Path
            p.mkdir(parents=True, exist_ok=True)

#######################
# DATABASE CONFIG     #
#######################

@dataclass
class DatabaseConfig:
    """Конфигурация подключений к базам данных."""
    
    # PostgreSQL параметры
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: str = os.getenv("POSTGRES_PORT", "5432")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "password")
    postgres_db: str = os.getenv("POSTGRES_DB", "mydatabase")
    
    # ClickHouse параметры
    clickhouse_host: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    clickhouse_port: str = os.getenv("CLICKHOUSE_PORT", "8123")
    clickhouse_user: str = os.getenv("CLICKHOUSE_USER", "default")
    clickhouse_password: str = os.getenv("CLICKHOUSE_PASSWORD", "password")
    clickhouse_db: str = os.getenv("CLICKHOUSE_DB", "default")
    clickhouse_protocol: str = os.getenv("CLICKHOUSE_PROTOCOL", "http")  # 'http' or 'https'
    
    # ClickHouse IPDR параметры
    clickhouse_ipdr_host: str = os.getenv("CLICKHOUSE_IPDR_HOST", "localhost")
    clickhouse_ipdr_user: str = os.getenv("CLICKHOUSE_IPDR_USER", "default")
    clickhouse_ipdr_password: str = os.getenv("CLICKHOUSE_IPDR_PASSWORD", "password")
    clickhouse_ipdr_db: str = os.getenv("CLICKHOUSE_IPDR_DB", "ipdr")
    
    # SSL настройки
    ca_path: str = "/data/azhusupbekova/bee_skymobile_local_dmp_ca.crt"
    
    def __post_init__(self):
        """Инициализация после создания объекта."""
        self.ssl_args = {'ssl_ca': self.ca_path, "verify": False}
    
    @property
    def postgres_url(self) -> str:
        """URL подключения к PostgreSQL."""
        return (f"postgresql://{self.postgres_user}:"
                f"{quote_plus(self.postgres_password)}@"
                f"{self.postgres_host}:{self.postgres_port}/"
                f"{self.postgres_db}")
    
    @property
    def clickhouse_url(self) -> str:
        """URL подключения к ClickHouse."""
        # ИСПРАВЛЕНО: формат как в рабочем примере
        url = (f"clickhouse://"
               f"{self.clickhouse_user}:{quote_plus(self.clickhouse_password)}@"
               f"{self.clickhouse_host}:{self.clickhouse_port}/"
               f"{self.clickhouse_db}?protocol={self.clickhouse_protocol}")
        return url
    
    @property
    def clickhouse_ipdr_url(self) -> str:
        """URL подключения к ClickHouse IPDR."""
        return (f"clickhouse://"
                f"{self.clickhouse_ipdr_user}:{quote_plus(self.clickhouse_ipdr_password)}@"
                f"{self.clickhouse_ipdr_host}/{self.clickhouse_ipdr_db}")

#######################
# FEATURE CONFIG      #
#######################
@dataclass
class FeatureConfig:
    """Конфигурация параметров обработки признаков."""
    NAME_DATAFRAME: str = 'features_weeks'    # имя файла с признаками
    DATE_FEATURES: str = '2025-11-05'         # дата формирования датасета
    SPLINT_DATE: str = '2024-12-01'           # дата сплита на трейн и валидацию
    DATE_END: str = '2025-11-04'              # cap for date_open (filtering); set as needed
    TARGET_COL: str = 'target'                # колонка целевая переменная
    DROP_FEATURES: list = field(default_factory=lambda: ['CELL_ID', 'CELL_MAX', 'DEV_NAME', 'TAC'])  # признаки для удаления перед обучением модели
    COUNT_WEEKS: int = 12                    # ожидаем 1..12, где 1 ближе всего к дате выдачи (должен быть последним тайм-степом)
    OVERDUE_DAYS_MAX: int = 30                # пример порога для 'плохой' целевой переменной
    TOTAL_OVERDUE: int = 90                    # пример порога для 'плохой' целевой переменной
    CURRENT_DATE: str = datetime.now().strftime('%Y-%m-%d')  # текущая дата в формате 'YYYY-MM-DD'
    NAME_DATAFRAME_WEEKS: str = f'{NAME_DATAFRAME}_{COUNT_WEEKS}_{OVERDUE_DAYS_MAX}_{TOTAL_OVERDUE}' # имя файла с признаками по неделям for LSTMs models
    KEYS_COLUMNS: list = field(default_factory=lambda: ['days_from_dt_end_to_price_change_date', 'FLAG_DEVICE_4G', 'days_from_dt_end_to_date_lad', 'USAGE_INTERNET_NIGHT', 
                               'ACTIVE_IND', 'REGION_CELL', 'days_from_dt_end_to_act_date', 'REVENUE_ABONKA', 'GENDER', 'BALANCE_END', 'USAGE_INTERNET_LTE', 
                               'COUNT_RECHARGE', 'LIFETIME_TOTAL', 'USAGE_ABONKA_TP', 'INTERCONNECT_MN_IN', 'USAGE_INTERNET_3G_FREE', 'days_from_dt_end_to_date_contract', 
                               'USAGE_NUM_INTERNET_PAK', 'REVENUE_INTERNET_PAYG', 'USAGE_OUT_INT_VOICE_RUSSIA'])  # используемые ключевые колонки для модели
    SEQ_COL: str = 'count_weeks'  # имя колонки с последовательностями для LSTM моделей

#######################
# LSTM CONFIG         #
#######################
@dataclass
class LSTMConfig:
    """Конфигурация параметров LSTM модели."""
    CALIBRATION_METHOD: str = 'sigmoid'  # метод калибровки: 'isotonic' или 'sigmoid'
    DROPOUT: float = 0.3  # дропаут в LSTM. Увеличим для борьбы с переобучением
    BIDIR: bool = False  # двунаправленная LSTM
    HIDDEN: int = 128  # размер скрытого состояния LSTM
    NUM_LAYERS: int = 1           # ← FIX: 2 слоя вместо 12 (лучше для стабильности)
    LEARNING_RATE: float = 1e-4      # 0.0001
    WEIGHT_DECAY: float = 1e-3       # L2 регуляризация 0.001
    NUM_EPOCHS: int = 50           # максимум эпох
    PATIENCE: int = 15              # early stopping по валидации
    BATCH_SIZE: int = 64             # Размер батча
#######################
# MLFLOW CONFIG       #
#######################


@dataclass
class MLflowConfig:
    """Конфигурация параметров MLflow."""
    NAME_PROJECT: str = 'scoring_eldik'
    HOST_MLFLOW: str = "http://10.16.230.222:5000"
    NAME_MODEL_CLIENT: str = 'LSTM_Scoring'
    EXPERIMENT_DESCRIPTION: str = """Модель для скоринга клиентов на основе поведенческих и демографических данных. """
    # Настройка клиента S3 (убедитесь, что credentials настроены)
    MINIO_CONSOLE_PORT: int = os.getenv('MINIO_CONSOLE_PORT', 9023)
    MINIO_PORT: int = os.getenv('MINIO_PORT', 9022)
    MINIO_ENDPOINT: str = os.getenv('MINIO_ENDPOINT', 'http://10.16.230.222:9022')
    ARTIFACT_ROOT: str = os.getenv('ARTIFACT_ROOT', 's3://mlflow/')
    BUCKET_NAME: str = os.getenv('BUCKET_NAME', 'mlflow')
    MLFLOW_S3_ENDPOINT_URL: str = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://10.16.230.222:9022')
    AWS_ACCESS_KEY_ID: str = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
    AWS_SECRET_ACCESS_KEY: str = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')

#######################
# MAIN CONFIG         #
#######################

class Config:
    """Главный конфигурационный класс."""
    
    def __init__(self):
        self.environment = EnvironmentConfig()
        self.database = DatabaseConfig()
        self.features = FeatureConfig()
        self.lstm = LSTMConfig()
        self.mlflow_config = MLflowConfig()
        
        # Создаем директории
        self.environment.create_directories()

# Создаем глобальный экземпляр
config = Config()

print("Configuration loaded successfully.")