import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from tqdm import tqdm
from src.logger import logger
from joblib import Parallel, delayed
import copy

# Функция для расчета целевой метрики
def objective(train_metric, test_metric, BETA):
    """
    Функция для расчета целевой метрики на основе тренировочной и тестовой метрик.
    
    Аргументы:
    train_metric -- метрика на обучающей выборке.
    test_metric -- метрика на тестовой выборке.
    BETA -- коэффициент экспоненциальной функции.
    
    Возвращает:
    Значение целевой метрики.
    """
    return -test_metric * np.exp(-BETA * abs(test_metric - train_metric))

# Функция для оценки модели с использованием CatBoost
def score(params, ROUNDS_MIN, N_SPLITS, RANDOM_STATE, categorical_features, PATIENCE, X, y, BETA):
    """
    Оценка модели с использованием перекрестной валидации и метрики ROC-AUC.
    
    Аргументы:
    params -- гиперпараметры для модели CatBoostClassifier.
    ROUNDS_MIN -- минимальное количество итераций для CatBoost.
    N_SPLITS -- количество фолдов для StratifiedKFold.
    RANDOM_STATE -- случайное состояние для воспроизводимости.
    categorical_features -- список категориальных признаков.
    PATIENCE -- количество итераций для ранней остановки.
    X -- данные для обучения.
    y -- целевая переменная.
    BETA -- коэффициент экспоненциальной функции для расчета целевой метрики.
    
    Возвращает:
    log -- словарь с результатами оценки модели.
    """
    metrics = []
    i_metrics = []
    models = []
    best_iteration = []

    # Стратегия перекрестной валидации
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    logger.info(f"Начало оценки модели с параметрами: {params}")
    logger.info(f"Кол-во фолдов для перекрестной валидации: {N_SPLITS}")

    # Обучение на каждом фолде
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        logger.info(f"Начало обучения на фолде {fold}/{N_SPLITS}")

        x_train, x_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        train_pool = Pool(data=x_train, label=y_train_fold, cat_features=categorical_features)
        test_pool = Pool(data=x_test, label=y_test_fold, cat_features=categorical_features)

        # Инициализация и обучение модели
        model = CatBoostClassifier(**params, iterations=ROUNDS_MIN, early_stopping_rounds=PATIENCE, random_seed=RANDOM_STATE)
        
        try:
            model.fit(train_pool, eval_set=test_pool, verbose=False)
        except Exception as e:
            logger.error(f"Ошибка при обучении модели на фолде {fold}: {e}")
            continue

        logger.info(f"Метрики для фолда {fold}: {model.best_score_}")

        # Прогнозы и расчёт метрик на тренировочной и тестовой выборках
        train_pred = model.predict_proba(x_train)[:, 1]
        test_pred = model.predict_proba(x_test)[:, 1]

        train_metric = roc_auc_score(y_train_fold, train_pred)
        test_metric = roc_auc_score(y_test_fold, test_pred)

        # Сохранение метрик и модели
        metrics.append((train_metric, test_metric))
        i_metrics.append((train_metric, test_metric, objective(train_metric, test_metric, BETA)))
        best_iteration.append(model.best_iteration_)

        models.append(model)
    
    # Средние метрики по всем фолдам
    if len(metrics) == 0:
        avg_metrics = [np.nan, np.nan]
    else:
        arr = np.array(metrics)
        if arr.ndim == 1:
            # metrics может быть плоским списком чисел или парой -> обрабатываем безопасно
            if arr.size == 2:
                avg_metrics = [float(arr[0]), float(arr[1])]
            else:
                avg_metrics = [float(arr.mean()), np.nan]
        else:
            avg_row = arr.mean(axis=0)
            if np.isscalar(avg_row):
                avg_metrics = [float(avg_row), np.nan]
            else:
                # явно приводим к float для стабильности
                avg_metrics = [float(avg_row[0]), float(avg_row[1])]
    
    log = {
        'params': params,
        'metrics': avg_metrics,
        'i_metrics': i_metrics,
        'best_iter': ROUNDS_MIN,
        'objective': objective(*avg_metrics, BETA),
        'models': models
    }

    logger.info(f"Завершение оценки модели. Средние метрики: {avg_metrics}")
    
    return log

def scoring(params, ROUNDS_MIN, BETA, X, y, N_SPLITS, RANDOM_STATE, categorical_features, PATIENCE, history, print_log=False):
    """
    Функция для оценки и оптимизации гиперпараметров модели.
    
    Аргументы:
    params -- гиперпараметры для модели CatBoostClassifier.
    history -- список для сохранения истории.
    ROUNDS_MIN -- минимальное количество итераций для CatBoost.
    BETA -- коэффициент экспоненциальной функции для расчета целевой метрики.
    X -- данные для обучения.
    y -- целевая переменная.
    N_SPLITS -- количество фолдов для StratifiedKFold.
    RANDOM_STATE -- случайное состояние для воспроизводимости.
    categorical_features -- список категориальных признаков.
    PATIENCE -- количество итераций для ранней остановки.
    print_log -- флаг для вывода логов.
    
    Возвращает:
    result -- результат оптимизации в формате, пригодном для использования с Hyperopt.
    """

    result = score(params, ROUNDS_MIN=ROUNDS_MIN, N_SPLITS=N_SPLITS, RANDOM_STATE=RANDOM_STATE, 
                   categorical_features=categorical_features, PATIENCE=PATIENCE, X=X, y=y, BETA=BETA)
        
    # Сохранение результатов в истории
    history.append(result)

    # Формирование лога для отладки
    log = f"""
        -----------------------------------
        Mean AUC train \t {result['metrics'][0]}
        Mean AUC test \t {result['metrics'][1]}
        Objective \t {-1 * objective(*result['metrics'], BETA)}
        Params \t {params}
        -----------------------------------
    """
    
    if print_log:
        print(log)
    
    # Формирование результата для Hyperopt
    return {
        'loss': objective(*result['metrics'], BETA),
        'status': STATUS_OK, 
        'mean_auc_train': result['metrics'][0], 
        'mean_auc_test': result['metrics'][1], 
        'best_iter': result['best_iter'],
        'params': params,
        'models': result['models']
    }

# Пространство гиперпараметров для Hyperopt
def optimize(trials, MAIN_METRIC, N_TRIALS, RANDOM_STATE, BETA, X, y, N_SPLITS, categorical_features, PATIENCE, ROUNDS_MIN):
    """
    Функция для настройки гиперпараметров модели с использованием Hyperopt.
    
    Аргументы:
    trials -- объект Hyperopt Trials для хранения результатов.
    MAIN_METRIC -- основная метрика для оценки модели.
    N_TRIALS -- количество итераций оптимизации.
    RANDOM_STATE -- случайное состояние для воспроизводимости.
    BETA -- коэффициент экспоненциальной функции для расчета целевой метрики.
    X -- данные для обучения.
    y -- целевая переменная.
    N_SPLITS -- количество фолдов для StratifiedKFold.
    categorical_features -- список категориальных признаков.
    PATIENCE -- количество итераций для ранней остановки.
    ROUNDS_MIN -- минимальное количество итераций для CatBoost.
    
    Возвращает:
    best -- наилучшие найденные гиперпараметры.
    history -- история всех итераций оптимизации.
    """
    history = []

    # Пространство гиперпараметров для перебора
    space = {
            # Глубина деревьев. Чем глубже дерево, тем более сложные зависимости оно может выучить.
            'depth': hp.choice('depth', list(range(1, 10))),
            # Количество итераций (деревьев) в модели. Увеличение этого параметра может улучшить модель, но также увеличит время обучения.    
            #'iterations': hp.quniform('iterations', 100, 1000, 50),
            # Скорость обучения. Низкие значения приводят к более плавному обучению, но требуют большего количества итераций.
            'learning_rate': hp.quniform('learning_rate', 0.05, 0.1, 0.025),
            # Регуляризация L2 для весов листьев дерева. Помогает избежать переобучения.
            'l2_leaf_reg': hp.quniform('l2_leaf_reg', 0, 5, 1),
            # Минимальное количество данных в листе. Увеличение этого значения может уменьшить переобучение, но сделать модель менее гибкой.
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 20, 1),
            # Температура для бутстрапирования. Влияет на случайность отбора подвыборок для обучения каждого дерева.
            'bagging_temperature': hp.quniform('bagging_temperature', 0.0, 0.5, 0.05),
            # Параметр для контроля уровня шума в данных. Большие значения могут улучшить устойчивость модели к шуму, но замедляют обучение.
            'random_strength': hp.quniform('random_strength', 1e-9, 1, 1e-9),
            # Количество границ для разбиения числовых признаков в бины. Большие значения могут повысить точность модели, но увеличивают время обучения.
            'border_count': hp.choice('border_count', [32, 64, 128, 254]),
            # Доля признаков, используемых для каждого уровня дерева. Меньшие значения могут уменьшить переобучение.
            'colsample_bylevel': hp.quniform('colsample_bylevel', 0.5, 1.0, 0.05),
            # Количество итераций для оценки веса листа. Чем больше итераций, тем более точная оценка, но дольше время обучения.
            'leaf_estimation_iterations': hp.choice('leaf_estimation_iterations', list(range(1, 20))),
            # Метод оценки весов листьев. "Newton" дает более точные результаты, но "Gradient" может быть быстрее.
            'leaf_estimation_method': hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),
            # Установка для выполнения задачи на GPU.
            'auto_class_weights': hp.choice('auto_class_weights', [None, 'Balanced', 'SqrtBalanced']),
            'task_type': 'CPU',
            'thread_count': 60,
            # Основная метрика для оценки модели.
            'eval_metric': MAIN_METRIC,
            # Уровень логирования. Установлен на 'Silent' для минимизации вывода логов.
            'logging_level': 'Silent'
        }
    # Оптимизация гиперпараметров
    best = fmin(
        fn=lambda p: scoring(p, ROUNDS_MIN=ROUNDS_MIN, BETA=BETA, X=X, y=y, N_SPLITS=N_SPLITS, 
                             RANDOM_STATE=RANDOM_STATE, categorical_features=categorical_features, 
                             PATIENCE=PATIENCE, history=history),
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=N_TRIALS,
        rstate=np.random.default_rng(RANDOM_STATE),
        show_progressbar=True  # Включите отображение прогресса
    )
    
    return best, history

# Функция для отображения важности признаков
def plot_feature_importance(model, feature_names, n_features=20):
    """
    Функция для построения графика важности признаков модели.
    
    Аргументы:
    model -- обученная модель CatBoostClassifier.
    feature_names -- список имен признаков.
    n_features -- количество признаков для отображения.
    """
    feature_importance = model.get_feature_importance()
    n_features = min(n_features, len(feature_importance))
    sorted_idx = np.argsort(feature_importance)[-n_features:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(n_features), feature_importance[sorted_idx], align='center')
    plt.yticks(range(n_features), np.array(feature_names)[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {n_features} Features")
    plt.show()
    return feature_importance

# Функция для отображения и выбора лучшего результата
def display_history_and_select_best(history):
    """
    Функция для отображения всей истории гиперпараметров и метрик, а также для выбора лучшего лога.
    
    Аргументы:
    history -- история всех итераций оптимизации.
    
    Возвращает:
    best_log -- наилучший найденный лог с метриками и параметрами.
    """
    print("Full hyperparameter optimization history:")
    for i, trial in enumerate(history, 1):
        print(f"Trial {i}:")
        pprint(trial)
        print("\n")

    # Поиск наилучшего лога
    best_log = min(history, key=lambda x: x['objective'], default=None)
    
    if best_log:
        print("Best log found:")
        pprint(best_log)
        
        print("Detailed metrics per fold for the best log:")
        for index, mtrcs in enumerate(best_log['i_metrics']):
            print(f"Fold {index}: train = {mtrcs[0]}, test = {mtrcs[1]}, objective = {-mtrcs[2]}")
    else:
        print("No best log found.")
    
    return best_log

def select_best_model(best_log):
    """
    Функция для выбора лучшей модели на основе минимального значения objective из best_log.

    Аргументы:
    best_log -- лог с метриками, параметрами и моделями для разных Fold.

    Возвращает:
    best_model -- лучшая модель на основе минимального значения objective.
    best_fold_index -- индекс лучшего Fold.
    """
    # Инициализация переменных для поиска лучшего Fold
    best_objective = float('-inf')
    best_model = None
    best_fold_index = -1

    # Проход по всем метрикам для каждого Fold
    for index, metrics in enumerate(best_log['i_metrics']):
        if abs(metrics[2]) > best_objective:
            best_objective = abs(metrics[2])
            best_fold_index = index

    # Получаем соответствующую модель
    if best_fold_index != -1:
        best_model = best_log['models'][best_fold_index]
        logger.info(f"Выбрана лучшая модель для сохранения. Лучший objective: {best_objective}, Fold: {best_fold_index}")
    else:
        logger.warning("Не удалось найти лучшую модель.")

    return best_model, best_fold_index



def save_best_model(model_name, model_dir, model, importance_threshold, PATH_RAW, CLASS_CLIENTS):
    """
    Функция для сохранения лучшей модели, получения и нормализации важности признаков.

    Аргументы:
    model_name -- имя для сохранения модели.
    model_dir -- директория для сохранения модели.
    model -- обученная модель CatBoost.
    importance_threshold -- порог важности признаков для фильтрации.
    PATH_RAW -- путь для сохранения DataFrame с важностью признаков.

    Возвращает:
    df_gain -- DataFrame с важностью признаков, отсортированный по убыванию.
    model_file_path -- путь к сохраненной модели.
    """

    # Получаем важность признаков
    feature_names = model.feature_names_
    feature_importance = model.get_feature_importance()

    # Создаем DataFrame с важностью признаков
    df_gain = pd.DataFrame({
        'name': feature_names,
        'value': feature_importance
    })

    # Нормализуем значения важности, чтобы сумма была равна 1
    df_gain['value'] = round(df_gain['value'] / df_gain['value'].sum(), 4)

    # Сортируем DataFrame по значению важности
    df_gain = df_gain.sort_values(by='value', ascending=False).reset_index(drop=True)
    logger.info("Важность признаков рассчитана и нормализована.")

    # Отфильтровываем признаки с важностью выше порога
    df_gain = df_gain[df_gain['value'] >= importance_threshold]
    logger.info(f"Признаки с важностью выше {importance_threshold} отфильтрованы.")

    # Сохраняем модель
    model_file_path = f'{model_dir}/{model_name.lower().replace(" ", "_")}.cbm'
    model.save_model(model_file_path, format='cbm')
    logger.info(f"Модель сохранена по пути: {model_file_path}")

    # Сохраняем DataFrame с важностью признаков
    df_gain_path = f'{PATH_RAW}/df_gain_{CLASS_CLIENTS}.csv'
    df_gain.to_csv(df_gain_path, index=False)
    logger.info(f"Важность признаков сохранена: {df_gain_path}")

    return df_gain, model_file_path

def load_model_and_get_info(model_file_path):
    """
    Функция для загрузки модели CatBoost, получения ее параметров, 
    лучшего числа итераций и имен признаков.

    Аргументы:
    model_file_path -- путь к сохраненной модели.

    Возвращает:
    model -- загруженная модель CatBoostClassifier.
    best_params -- параметры модели.
    rounds_1stage -- лучшее число итераций.
    feature_1 -- список имен признаков.
    categorical_features -- список категориальных признаков.
    df_gain -- DataFrame с важностью признаков.
    """
    model = CatBoostClassifier()
    model.load_model(model_file_path, format='cbm')

    best_params = model.get_params()
    rounds_1stage = model.get_best_iteration()
    feature_names = model.feature_names_
    feature_importance = model.get_feature_importance()
    cat_feature_indices = model.get_cat_feature_indices()

    # Создаем DataFrame с важностью признаков
    df_gain = pd.DataFrame({
        'name': feature_names,
        'value': feature_importance
    })

    # Нормализуем значения важности, чтобы сумма была равна 1
    df_gain['value'] = round(df_gain['value'] / df_gain['value'].sum(), 4)
    df_gain.sort_values(by='value', ascending=False, inplace=True)

    categorical_features = [feature_names[i] for i in cat_feature_indices]

    logger.info(f'Параметры модели: {best_params}')
    logger.info(f'Лучшее число итераций: {rounds_1stage}')
    logger.info(f'Признаки: {feature_names}')
    logger.info(f'Индексы категориальных признаков: {cat_feature_indices}')
    logger.info(f'Названия категориальных признаков: {categorical_features}')
    # Удалите несколько ключей из best_params, если они существуют
    keys_to_remove = ['iterations', 'verbose', 'logger_level', 'verbose_eval', 'silent']
    for key in keys_to_remove:
        best_params.pop(key, None)  # pop(key, None) удаляет ключ, если он существует, игнорируя ошибку, если нет

    
    return model, best_params, rounds_1stage, feature_names, categorical_features, df_gain



def feature_selection_and_evaluation(df_gain, X_train, X_valid, y_train, y_valid, best_params, rounds_1stage, categorical_features, BETA, dk, k, n_jobs=None):
    """
    Параллельная версия: для каждого k (кол-во признаков) запускает обучение в отдельном процессе.
    n_jobs -- число параллельных процессов (None -> joblib решит, -1 -> все CPU).
    Возвращает (f, final_model)
    final_model обучается последовательно на полном наборе признаков (k финальный).
    """
    df_gain = df_gain.sort_values(by='value', ascending=False).reset_index(drop=True)

    ks = list(range(k, df_gain.shape[0] + 1, dk))
    if len(ks) == 0:
        return pd.DataFrame(columns=['Number of columns', 'ROC AUC TEST', 'ROC AUC TRAIN', 'objective']), None

    # Вспомогательная функция, выполняемая в процессе
    def _train_and_eval(k_val):
        # локальная копия params — работа в отдельном процессе
        params = copy.deepcopy(best_params) if isinstance(best_params, dict) else dict(best_params)
        # избегаем nested threading — для воркера ставим 1 поток
        params['thread_count'] = 1
        # явное приведение типов
        params['depth'] = int(params.get('depth', 6)) if 'depth' in params else 6
        params['min_data_in_leaf'] = int(params.get('min_data_in_leaf', 1))
        params['leaf_estimation_iterations'] = int(params.get('leaf_estimation_iterations', 1)) if 'leaf_estimation_iterations' in params else params.get('leaf_estimation_iterations', 1)

        valid_columns = df_gain['name'].iloc[:k_val].tolist()
        valid_categorical_features = [col for col in categorical_features if col in valid_columns]

        train_pool = Pool(data=X_train[valid_columns], label=y_train, cat_features=valid_categorical_features)
        test_pool = Pool(data=X_valid[valid_columns], label=y_valid, cat_features=valid_categorical_features)

        model_local = CatBoostClassifier(**params, iterations=int(rounds_1stage))
        try:
            model_local.fit(train_pool, eval_set=test_pool, verbose=False)
            train_pred = model_local.predict_proba(X_train[valid_columns])[:, 1]
            test_pred = model_local.predict_proba(X_valid[valid_columns])[:, 1]
            train_metric = roc_auc_score(y_train, train_pred)
            test_metric = roc_auc_score(y_valid, test_pred)
            obj = -objective(train_metric, test_metric, BETA)
        except Exception as e:
            # в случае ошибки возвращаем NaN и лог
            logger.error(f"Error training k={k_val}: {e}")
            train_metric = np.nan
            test_metric = np.nan
            obj = np.nan

        return {'k': k_val, 'test': test_metric, 'train': train_metric, 'objective': obj, 'cols': valid_columns}

    # Запускаем параллельно (tqdm вокруг генератора чтобы видеть прогресс)
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_train_and_eval)(k_val) for k_val in tqdm(ks)
    )

    # Собираем DataFrame в нужном порядке
    f = pd.DataFrame(results).sort_values(by='k').reset_index(drop=True)
    f = f.rename(columns={'k': 'Number of columns', 'test': 'ROC AUC TEST', 'train': 'ROC AUC TRAIN', 'objective': 'objective', 'cols': 'Columns'})

    return f

