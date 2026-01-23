# Получение предсказаний по модели airflow для кредитного скоринга и загрузка в ClickHouse

# %% [markdown]
## Импортируем необходимые библиотеки

import pandas as pd
import sys
import mlflow
import torch

from torch.utils.data import DataLoader
import boto3
import io
from pathlib import Path
from datetime import date, timedelta, datetime
from mlflow.tracking import MlflowClient
import ast
import joblib
from sqlalchemy import inspect
# --- Настройка путей и sys.path ---
# Добавляем корневую директорию проекта в sys.path для импорта кастомных модулей
PROJECT_ROOT = Path().cwd().parent
sys.path.append(str(PROJECT_ROOT))

from src.config import config
from src.logger import logger
from src.database import clickhouse_engine
from src.visualization import *
from src.predprocessing_lstm import create_lstm_sequences_credit_scoring, convert_categorical_to_str, collate_fn
from src.base_models import LSTMCreditScoringDataset, InsertPredictionsScoring, Base
from src.modeling_lstm import _predict_probs_from_loader
from sqlalchemy.orm import sessionmaker

# %%
# --- Настройка устройства для вычислений ---.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
# %% [markdown]
## Mlflow настройка и получение артефактов модели
logger.info("MLflow client created and tracking URI set.")
mlflow.set_tracking_uri(config.mlflow_config.HOST_MLFLOW)
client = MlflowClient()
exp = client.get_experiment_by_name(config.mlflow_config.NAME_PROJECT)
# сделать эксперимент активным
mlflow.set_experiment(config.mlflow_config.NAME_PROJECT)
EXP_ID = mlflow.get_experiment_by_name(config.mlflow_config.NAME_PROJECT).experiment_id
logger.info(f"Experiment ID: {EXP_ID}")

version_info = client.get_model_version_by_alias(name=config.mlflow_config.NAME_MODEL_CLIENT, alias='test')
logger.info(f"Model version info for alias 'test': {version_info}")
RUN_ID = version_info.run_id
logger.info(f"Найден run_id по алиасу 'test': {RUN_ID}")

SPLINT_DATE = client.get_run(RUN_ID).data.params.get('split_date')
COUNT_WEEKS = int(client.get_run(RUN_ID).data.params.get('count_weeks'))
logger.info(f"COUNT_WEEKS: {COUNT_WEEKS}")
DATE_FEATURES = client.get_run(RUN_ID).data.params.get('date_features')
KEYS_COLUMNS = ast.literal_eval(client.get_run(RUN_ID).data.params.get('name_columns'))
logger.info(f"Ключевые колонки для загрузки из DWH: {KEYS_COLUMNS}")



# %%
def load_from_s3(bucket, s3_client, key):
    '''Загружает файл из S3 и возвращает его как BytesIO объект'''
    response = s3_client.get_object(Bucket=bucket, Key=key)
    data = response['Body'].read()
    return io.BytesIO(data)

def get_from_s3_sklearn_artifacts(exp_id, run_id):
    '''Загрузка артефактов sklearn модели из S3 (препроцессор, колонки и т.д.)'''
    s3_client = boto3.client(
        's3',
        endpoint_url=config.mlflow_config.MINIO_ENDPOINT,
        aws_access_key_id=config.mlflow_config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.mlflow_config.AWS_SECRET_ACCESS_KEY,
    )

    # Загрузка файлов напрямую
    prefix = f"{exp_id}/{run_id}/artifacts/preprocessor"

    cat_maps = joblib.load(load_from_s3(config.mlflow_config.BUCKET_NAME, s3_client, f"{prefix}/cat_maps.pkl"))
    preprocessor = joblib.load(load_from_s3(config.mlflow_config.BUCKET_NAME, s3_client, f"{prefix}/preprocessor_lstm_{DATE_FEATURES}.joblib"))

    numeric_cols = joblib.load(load_from_s3(config.mlflow_config.BUCKET_NAME, s3_client, f"{prefix}/numeric_cols.pkl"))
    categorical_cols = joblib.load(load_from_s3(config.mlflow_config.BUCKET_NAME, s3_client, f"{prefix}/categorical_cols.pkl"))
    logger.info(f"Numeric cols: {len(numeric_cols)}, categorical cols: {len(categorical_cols)}")

    model_key = version_info.source.replace(f's3://{config.mlflow_config.BUCKET_NAME}/', '') + '/data/model.pth'

    response = s3_client.get_object(Bucket=config.mlflow_config.BUCKET_NAME, Key=model_key)
    model_buffer = io.BytesIO(response['Body'].read())
    model = torch.load(model_buffer, map_location=DEVICE, weights_only=False)
    logger.info(f"Loaded model: {model}")
    return preprocessor, cat_maps, numeric_cols, categorical_cols, model

def make_query(engine, number_weeks, current_date):
    """
    Возвращает только нужные KEYS_COLUMNS из DWH.dm_datamart_weekly.
    Для days_from_dt_end_to_* считаем: dateDiff('day', <DATE_COL>, addDays(DT, 7))
    """
    computed = {
        'days_from_dt_end_to_price_change_date': "coalesce(dateDiff('day', PRICE_CHANGE_DATE, addDays(DT, 7)), -1) AS days_from_dt_end_to_price_change_date",
        'days_from_dt_end_to_act_date':          "coalesce(dateDiff('day', ACT_DATE,          addDays(DT, 7)), -1) AS days_from_dt_end_to_act_date",
        'days_from_dt_end_to_date_contract':     "coalesce(dateDiff('day', DATE_CONTRACT,     addDays(DT, 7)), -1) AS days_from_dt_end_to_date_contract",
        'days_from_dt_end_to_date_lad':          "coalesce(dateDiff('day', DATE_LAD,          addDays(DT, 7)), -1) AS days_from_dt_end_to_date_lad",
        'days_from_dt_end_to_date_inactive':      "coalesce(dateDiff('day', DATE_INACTIVE,     addDays(DT, 7)), -1) AS days_from_dt_end_to_date_inactive",
        'days_from_dt_end_to_date_abonka':        "coalesce(dateDiff('day', DATE_ABONKA,       addDays(DT, 7)), -1) AS days_from_dt_end_to_date_abonka",
    
    }

    select_items = []
    for col in KEYS_COLUMNS:  # KEYS_COLUMNS уже загружается выше из MLflow
        if col in computed:
            select_items.append(computed[col])
        else:
            select_items.append(col)
    select_items.append('CTN as ctn, SUBS_ID as subs_id')  # Добавляем колонку DT для фильтрации по дате
    select_clause = ",\n                ".join(select_items)
    query = f"""
        SELECT
                {select_clause}
        FROM DWH.dm_datamart_weekly w
        where w.DT <= toDate('{current_date}')
          AND w.DT = toStartOfWeek('{current_date}' - INTERVAL {number_weeks} WEEK - INTERVAL 1 DAY, 1) 
    """

    df = pd.read_sql(query, engine)
    logger.info(f"Loaded data for {number_weeks} weeks: {df.shape}")
    return df

def make_features(clickhouse_engine, current_date):
    '''Получение признаков для модели из сырых данных по нескольким неделям'''
    df_features_parts = []
    for number_week in range(1, COUNT_WEEKS+1):  # от 1 до 12 недель включительно
        logger.info(f"COUNT_WEEKS = {number_week}")
        df_part = make_query(clickhouse_engine, number_weeks=number_week
                             , current_date=current_date)
        if df_part is None or df_part.empty:
            logger.warning(f"COUNT_WEEKS = {number_week}, пустой датафрейм, пропускаем")
            continue
        df_part[config.features.SEQ_COL] = number_week
        logger.info(f"COUNT_WEEKS = {number_week}, shape = {df_part.shape}")
        df_features_parts.append(df_part)
    df = pd.concat(df_features_parts, ignore_index=True)
    logger.info(f"Final features shape: {df.shape}")
    return df

def filter_subs_id_with_count_weeks_1(df):
    '''Получаем уникальные subs_id с count_weeks == 1 и оставляем только их для предсказаний'''
    sub_ids = df.query(f"{config.features.SEQ_COL} == 1")['subs_id'].unique()
    df = df[df['subs_id'].isin(sub_ids)].reset_index(drop=True)
    logger.info(f"After filtering by subs_id with count_weeks == 1, shape: {df.shape}")
    df.set_index(['subs_id', 'ctn', config.features.SEQ_COL], inplace=True)
    logger.info(f"After setting index, shape: {df.shape}")
    return df

def transform_data_for_prediction(df2, numeric_cols, categorical_cols, cat_maps, preprocessor):
    ''' Преобразование данных для предсказания
    1. Преобразование категориальных признаков в строки
    2. Создание ID колонок для группировки
    3. Создание LSTM последовательностей для валидации
    4. Создание DataLoader для валидации
    5. Получение предсказаний из модели
    '''
    val_tab_full = convert_categorical_to_str(df2, categorical_cols)
    # Определяем ID колонки для группировки
    id_cols_from_index = list(val_tab_full.index.names)
    id_cols = [col for col in id_cols_from_index if col not in [config.features.SEQ_COL, config.features.TARGET_COL]]
    logger.info(f"ID columns for grouping: {id_cols}")

    # Создаем последовательности для validation
    Xn_val, Xc_val, y_val_seq, val_metadata, val_len = create_lstm_sequences_credit_scoring(
        df=val_tab_full,
        id_cols=id_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_maps=cat_maps,
        scaler=preprocessor, 
        seq_col=config.features.SEQ_COL,
        target_col=config.features.TARGET_COL,
        only_prediction=True
    )

    val_dataset = LSTMCreditScoringDataset(Xn_val, Xc_val, y_val_seq, val_len, val_metadata)
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    # ← Проверяем типы данных в батче
    sample_batch = next(iter(DataLoader(val_dataset, batch_size=4)))
    logger.info(f"Sample batch dtypes:")
    logger.info(f"  X_num dtype: {sample_batch['X_num'].dtype}")
    logger.info(f"  X_cat dtype: {sample_batch['X_cat'].dtype}")
    logger.info(f"  y dtype: {sample_batch['y'].dtype}")
    logger.info(f"  y values: {sample_batch['y']}")

    va_loader_pred = DataLoader(
        val_dataset,
        batch_size=config.lstm.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    logger.info(f"Val loader: {len(va_loader_pred)} batches")
    return va_loader_pred, val_dataset

def create_prediction_records(val_dataset, val_probs, current_date):
    '''Создание записей для загрузки в БД из предсказаний и метаданных'''
    # Создание датафрейма с предсказаниями и необходимыми колонками для загрузки в БД
    meta_df_valid = val_dataset.metadata.copy().reset_index(drop=True)
    meta_df_valid['probability'] = val_probs.astype(float)
    meta_df_valid['insert_datetime'] = pd.to_datetime(current_date)
    # Добавляем колонку observation_period_end_date (текущая дата + 7 дней, т.е предсказания актуальны в течение недели + понедельник и вторник следующей недели)
    # Это надо потому что витрина считается во вторние к обеду за прошлую неделю
    # Для предсказаний надо брать последную дату вставки
    meta_df_valid['observation_period_end_date'] = (pd.to_datetime(current_date) + pd.Timedelta(days=9)).date()
    meta_df_valid['load_dt'] = pd.Timestamp.utcnow()
    meta_df_valid['subs_id'] = meta_df_valid['subs_id'].astype('int64')
    meta_df_valid['ctn'] = meta_df_valid['ctn'].astype('int64')

    records = meta_df_valid[['subs_id','ctn','insert_datetime','probability',
                            'observation_period_end_date','load_dt']].to_dict('records')
    return records

def insert_predictions_to_clickhouse(clickhouse_engine, records, current_date):
    '''Вставка предсказаний в ClickHouse'''
    if not inspect(clickhouse_engine).has_table(InsertPredictionsScoring.__tablename__, schema="data_science"):
        logger.info(f"Таблица {InsertPredictionsScoring.__tablename__} не найдена, создаем новую.")
        Base.metadata.create_all(clickhouse_engine)

    Session = sessionmaker(bind=clickhouse_engine)
    session = Session()
    try:
        session.execute(InsertPredictionsScoring.__table__.insert(), records)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Ошибка при вставке предсказаний")
        raise
    finally:
        session.close()

    # проверка
    check_df = pd.read_sql(
        "SELECT subs_id, ctn, probability, insert_datetime, observation_period_end_date "
        "FROM data_science.credit_scoring_predictions "
        f"WHERE insert_datetime = toDate('{current_date}') "
        "ORDER BY insert_datetime DESC LIMIT 5",
        clickhouse_engine
    )
    logger.info("Последние 5 записей в таблице credit_scoring_predictions:")
    logger.info(check_df)

def run(settlement_date=date.today().strftime('%Y-%m-%d')):
    ''' Запуск предсказания для airflow 
    передаем дату в формате 'YYYY-MM-DD'
    '''
    if settlement_date > SPLINT_DATE:
        logger.info(f"Текущая дата {settlement_date} больше даты сплита {SPLINT_DATE} - дата обучающей выборки")
    else:
        logger.warning(f"Текущая дата {settlement_date} меньше даты сплита {SPLINT_DATE} - дата обучающей выборки")
    # Теперь запустим все необходимые шаги
    preprocessor, cat_maps, numeric_cols, categorical_cols, model = get_from_s3_sklearn_artifacts(EXP_ID, RUN_ID)
    df1 = make_features(clickhouse_engine, current_date=settlement_date)
    df2 = filter_subs_id_with_count_weeks_1(df1)
    va_loader_pred, val_dataset = transform_data_for_prediction(df2, numeric_cols, categorical_cols, cat_maps, preprocessor)
    # Перед инференсом переводим модель в eval режим
    model.eval()
    with torch.no_grad():
        val_probs = _predict_probs_from_loader(va_loader_pred, model, DEVICE)
        assert len(val_probs) == len(val_dataset), "Несоответствие длины вероятностей и датасета"

    records = create_prediction_records(val_dataset, val_probs, current_date=settlement_date)
    insert_predictions_to_clickhouse(clickhouse_engine, records, settlement_date)   

# %% [markdown]
if __name__ == '__main__':
    if len(sys.argv) > 1:
        raw_arg = sys.argv[1]
        logger.info(f"Получен аргумент даты (raw): {raw_arg}")
        run(raw_arg)
    else:
        run()
