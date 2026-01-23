import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from src.logger import logger
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def _process_group(group_tuple, id_cols, numeric_cols, categorical_cols, cat_maps, 
                   scaler, seq_col, target_col, only_prediction=False):
    """
    Обрабатывает одну группу данных (для одного клиента).
    Эта функция будет выполняться в отдельном процессе.
    """
    group_key, group = group_tuple
    
    # Сортируем по времени
    group_sorted = group.sort_values(seq_col, ascending=False)
    
    # 1. Преобразование числовых признаков
    # Копируем, чтобы избежать гонки данных при работе с fillna
    group_to_process = group_sorted.copy()
    group_to_process[numeric_cols] = group_to_process[numeric_cols].fillna(0).astype(np.float32)
    num_features = scaler.transform(group_to_process[numeric_cols])
    
    # 2. Преобразование категориальных признаков
    cat_features = []
    for col in categorical_cols:
        s = group_sorted[col].astype(str) # group_sorted здесь безопасен, т.к. нет inplace-изменений
        m = cat_maps[col]
        enc = s.map(m).fillna(m.get("missing", 0)).astype(np.int64).to_numpy()
        cat_features.append(enc)
    cat_features = np.stack(cat_features, axis=1)

    # 3. Целевая переменная
    y_target = group_sorted[target_col].iloc[0] if only_prediction == False else 0.0
    
    # 4. Метаданные
    metadata_dict = dict(zip(id_cols, group_key if isinstance(group_key, tuple) else (group_key,)))
    
    return (
        torch.from_numpy(num_features),
        torch.from_numpy(cat_features),
        y_target,
        len(group_sorted),
        metadata_dict
    )


def create_lstm_sequences_credit_scoring(df, id_cols, numeric_cols, categorical_cols, 
                                         cat_maps, scaler, seq_col='count_weeks', 
                                         target_col='target', only_prediction=False):
    """
    Создаёт LSTM последовательности в параллельном режиме, разделяя на числовые и категориальные.
    Применяет scaler для числовых и cat_maps для категориальных признаков.
    Поддерживает последовательности переменной длины.
    """
    df_reset = df.reset_index()
    groups = list(df_reset.groupby(id_cols))
    
    logger.info(f"Найдено {len(groups)} уникальных групп для обработки на нескольких ядрах.")
    
    # Параллельный запуск обработки групп
    results = Parallel(n_jobs=-1)(
        delayed(_process_group)(
            g, id_cols, numeric_cols, categorical_cols, 
            cat_maps, scaler, seq_col, target_col, only_prediction
        ) for g in tqdm(groups, desc="Создание последовательностей")
    )
    
    # Собираем результаты из всех процессов
    Xn_list, Xc_list, y_list, len_list, metadata_list = zip(*results)

    y = np.array(y_list, dtype=np.float32)
    metadata = pd.DataFrame(list(metadata_list))
    
    logger.info(f"Созданы списки последовательностей: {len(Xn_list)} шт.")
    
    # Возвращаем списки тензоров и длин
    return list(Xn_list), list(Xc_list), y, metadata, list(len_list)

def convert_categorical_to_str(df, categorical_cols):
    ''' Приводит категориальные колонки к типу str в переданном датафрейме '''
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    logger.info(f"Categorical columns converted to str")
    return df


def collate_fn(batch):
    """
    Собирает батч из последовательностей разной длины, применяя паддинг.
    """
    batch.sort(key=lambda x: x['len'], reverse=True)
    indices = torch.tensor([item['idx'] for item in batch], dtype=torch.long)
    X_num_list = [item['X_num'] for item in batch]
    X_cat_list = [item['X_cat'] for item in batch]
    ys = torch.stack([item['y'] for item in batch])
    lengths = torch.tensor([item['len'] for item in batch])
    X_num_padded = pad_sequence(X_num_list, batch_first=True, padding_value=0.0)
    X_cat_padded = pad_sequence(X_cat_list, batch_first=True, padding_value=0)
    return {
        'X_num': X_num_padded,
        'X_cat': X_cat_padded,
        'y': ys,
        'lengths': lengths,
        'indices': indices,
    }