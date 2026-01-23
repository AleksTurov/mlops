import pandas as pd
from tqdm import tqdm
from src.logger import logger
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score

def nan_cutoff(data, norm):
    """
    Убирает признаки с процентом пустых значений выше norm
    """
    nans_columns = [col for col in tqdm(data.columns) if data[col].isna().mean() > norm]
    data_reduced = data.drop(nans_columns, axis=1)
    logger.info(f'Dropped columns due to NaNs: {len(nans_columns)}')
    return data_reduced

def one_value_cutoff(data):
    """
    Убирает признаки с единственным значением
    """
    one_cols = [col for col in data.columns if data[col].nunique() < 2]
    data_reduced = data.drop(one_cols, axis=1)
    logger.info(f'Dropped columns with single unique value: {len(one_cols)}')
    logger.info(f'Single unique value columns: {one_cols}')
    return data_reduced

def convert_low_cardinality_to_category(data, threshold):
    """
    Преобразует признаки с числом уникальных значений меньше threshold в категориальные.
    Преобразует все категориальные признаки в строки.
    """
    for col in data.columns:
        if data[col].nunique() < threshold:
            data[col] = data[col].astype('category')
            data[col] = data[col].astype(str)  # Преобразование в строку
    logger.info(f'Converted low cardinality columns to categorical type')
    return data

def load_and_preprocess_data(path, DROP_FEATURES, NORM, THRESHOLD_UNIQUE, TARGET, del_col_flag=True, categorical_features=None):
    """
    Функция для загрузки и предобработки данных
    """
    # Шаг 1: Загрузка данных из CSV-файла
    # Шаг 1: Загрузка данных из CSV-файла
    try:
        data = pd.read_csv(path)
    except:
        try:
            data = pd.read_parquet(path)
        except Exception as e:
            logger.error(f'Error loading data from {path}: {e}')
            return None
    if DROP_FEATURES is None:
        DROP_FEATURES = []
    else:
        # Шаг 2: Удаление ненужных столбцов
        data.drop(DROP_FEATURES, axis=1, inplace=True)
    if del_col_flag:
        
        # Шаг 3: Удаление колонок с большим количеством NaN
        data = nan_cutoff(data, NORM)
        
        # Шаг 4: Удаление колонок с одним уникальным значением
        data = one_value_cutoff(data)
    if categorical_features is None:
        # Шаг 5: Преобразование признаков с малым числом уникальных значений в категориальные
        data = convert_low_cardinality_to_category(data, THRESHOLD_UNIQUE)
        
        # Шаг 6: Разделение признаков на числовые и категориальные
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
        
        if TARGET in categorical_features:
            categorical_features.remove(TARGET)
        # Шаг 7: Обработка пропусков в категориальных признаках
        for col in categorical_features:
            data[col] = data[col].astype('category')  # Преобразование в категориальный тип
            data[col] = data[col].cat.add_categories('NaN')  # Добавление новой категории 'NaN'
            data[col] = data[col].fillna('NaN')  # Заполнение пропусков значением 'NaN'
            data[col] = data[col].astype(str)  # Преобразование в строку
    else:
        for col in categorical_features:
            data[col] = data[col].fillna('NaN')  # Заполнение пропусков значением 'NaN'
            data[col] = data[col].astype(str)  # Преобразование в строку
    ## Шаг 8: Обработка пропусков в числовых признаках
    #for col in numeric_features:
    #    if col not in [TARGET]:
    #        data[f'{col}_fillna'] = data[col].fillna(-1)
    #        data[f'{col}_zero'] = data[col].fillna(0)
    #        data.drop(columns=[col], inplace=True)

    # Шаг 8: Определение признаков и целевой переменной
    features = [col for col in data.columns if col not in [TARGET] + DROP_FEATURES]
    X = data[features]
    y = data[TARGET]
    
    return X, y, features, categorical_features

def process_and_split_data(df, feature_names, categorical_features, target_col, test_size, random_state):
    """
    Функция для обработки данных и разделения на обучающую и тестовую выборки.

    Аргументы:
    df -- DataFrame с данными.
    feature_names -- список признаков.
    categorical_features -- список категориальных признаков.
    target_col -- название целевой переменной.
    test_size -- доля тестовой выборки.
    random_state -- случайное состояние для воспроизводимости.

    """
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in categorical_features:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.add_categories('NaN')
        df[col] = df[col].fillna('NaN')
        df[col] = df[col].astype(str)
    
    # Шаг 8: Обработка пропусков в числовых признаках
    #for col in numeric_features:
    #    if col not in [target_col]:
    #        data[f'{col}_fillna'] = data[col].fillna(-1)
    #        data[f'{col}_zero'] = data[col].fillna(0)
    #        data.drop(columns=[col], inplace=True)

    X = df[feature_names]
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    logger.info(f'Размер исходного датасета: {X.shape}, {y.shape}')
    logger.info(f'Размер обучающей выборки: {X_train.shape}, {y_train.shape}')
    logger.info(f'Размер валидационной выборки: {X_valid.shape}, {y_valid.shape}')

    return X_train, X_valid, y_train, y_valid, X, y

def drop_features_func(path_gain, DROP_FEATURES, path_data):
    '''
    Функция для определения лишних колонок
    path_gain - путь к файлу с важностью признаков
    DROP_FEATURES - список изначально лишних признаков
    path_data - путь к файлу с данными
    
    '''
    df_gain = pd.read_csv(path_gain)
    columns_name = df_gain['name'].unique()
    try:
        train = pd.read_csv(path_data)
    except:
        try:
            train = pd.read_parquet(path_data)
        except Exception as e:
            logger.error(f'Error loading data from {path_data}: {e}')
            return DROP_FEATURES

    # Найдите все колонки, которых нет в columns_name
    drop_columns = [col for col in train.columns if col not in columns_name]

    # Удалите 'target' из drop_columns
    if 'target' in drop_columns:
        drop_columns.remove('target')

    # Добавляем записи из drop_columns к DROP_FEATURES
    DROP_FEATURES.extend(drop_columns)
    return DROP_FEATURES