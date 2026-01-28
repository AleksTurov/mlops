# Архитектура и поток данных (RU)

Ниже — максимально подробное описание архитектуры, потоков данных и эксплуатации. Документ рассчитан на пользователя, который запускает стек впервые и хочет понимать, что где происходит, как проверять корректность и куда смотреть при ошибках.

---

## 1) Общая идея
Стек организован как **минимальный MLOps‑контур**:
- MLflow хранит эксперименты, метрики и registry моделей.
- MinIO хранит артефакты моделей (S3‑совместимое хранилище).
- Airflow выполняет пакетные задачи (данные, обучение, batch‑инференс).
- Автоматический сервинг строится на **MLflow Serve** контейнерах, которые поднимаются по alias.
- Наблюдаемость обеспечивается Prometheus + Grafana + Loki.

Цель — быстро получить **сквозную историю**: данные → обучение → alias → сервинг → мониторинг.

---

## 2) Компоненты и роли

### 2.1 MLflow и хранилища
- **MLflow** — трекинг экспериментов, реестр моделей, alias‑ы.
- **PostgreSQL (mlflow-db)** — метаданные MLflow (runs, metrics, registry, aliases).
- **MinIO** — артефакты моделей и вспомогательные файлы.

### 2.2 Airflow и данные
- **Airflow** — пайплайны (DAG‑и) для данных и обучения.
- **PostgreSQL (app-db)** — таблицы датасетов и предсказаний.

### 2.3 Сервинг моделей
- **MLflow Autoserve** — сервис‑наблюдатель за registry, поднимает `mlflow models serve` по alias.
- **MLflow Serve контейнеры** — отдельный контейнер на каждый alias (например, `iris_classifier_iris@Production`).

### 2.4 Наблюдаемость
- **Prometheus** — сбор метрик доступности.
- **Grafana** — визуализация и алерты.
- **Loki/Promtail** — сбор Docker‑логов и поиск в Grafana.
- **Blackbox exporter** — HTTP health‑checks, по которым строятся статусы.

---

## 3) Потоки данных (end‑to‑end)

### 3.1 Загрузка данных
**DAG:** `dag_data_predictions`
- Загружает тестовый датасет (например, iris) в **app-db**.
- На выходе в БД появляются таблицы с данными.

### 3.2 Обучение
**DAG:** `dag_training`
- Запускает `ml.training.train_candidate()`.
- Логирует параметры/метрики в MLflow.
- Сохраняет артефакты в MinIO.
- Регистрирует модель в MLflow Registry.
- Автоматически присваивает alias `Production` лучшей версии.

### 3.3 Сервинг по alias
**Сервис:** `mlflow-autoserve`
- Сканирует registry и aliases.
- Для каждого alias поднимает `mlflow models serve` контейнер.
- При смене alias контейнер пересоздаётся на новую версию.

### 3.4 Batch‑инференс
**DAG:** `dag_inference`
- Загружает данные из app-db.
- Вызывает модель (через MLflow) и пишет предсказания в app-db.

### 3.5 Мониторинг качества
**DAG:** `dag_model_monitoring`
- Сравнивает candidate vs production.
- Пишет метрики сравнения в MLflow.

---

## 4) Механика MLflow Autoserve

### 4.1 Что делает autoserve
- Читает список моделей и их alias в MLflow Registry.
- Для каждого alias определяет конкретную версию модели.
- **Прод‑режим:** строит отдельный Docker‑image на версию модели и запускает контейнер `mlflow-serve-<model>-<alias>` из этого image.
- Устанавливает метки контейнера: `mlflow_model`, `mlflow_alias`, `mlflow_version`, `mlflow_image`.

### 4.2 Как работает build per model version image
- Для каждой версии модели формируется image вида `mlflow-model-<model>-v<version>`.
- Image собирается через MLflow (`build_docker`), поэтому зависимости фиксируются **внутри image**.
- При смене alias на другую версию контейнер пересоздаётся на новый image.

### 4.3 API MLflow Serve
- `GET /ping` — проверка живости.
- `POST /invocations` — инференс (MLflow scoring format).

### 4.4 Контракт входа
Артефакты модели содержат:
- `data_contract/input_schema.json` — список признаков.
- `data_contract/sample_input.csv` — пример входа.
- `metrics/validation_metrics.json` — итоговые метрики.

---

## 5) Наблюдаемость (детально)

### 5.1 Service Health
**Grafana → Service Health Detailed**
- Показывает `MLflow`, `Airflow`, `MinIO`, `Prometheus`, `Loki`, `Grafana`.
- Показывает `MLflow Serve (aliases)` по `model@alias`.

### 5.2 MLflow Serving dashboard
**Grafana → MLflow Serving**
- Статус alias по `probe_success`.
- Latency (`probe_duration_seconds`).

### 5.3 Логи
**Grafana Explore (Loki)**
- Фильтры: `container`, `service`, `project=mlops`.
- Используется для диагностики проблем с DAG‑ами и сервингом.

### 5.4 Почему нет /metrics
MLflow Serve не предоставляет Prometheus `/metrics`, поэтому доступны только health‑checks через Blackbox.
Если нужны RPS/latency/errors — требуется внешний gateway.

---

## 6) Авто‑демо и первый запуск
```bash
cp env.dev.example .env
docker compose --env-file .env up -d --build
```
Контейнер `demo-bootstrap`:
- ждёт готовности MLflow и Airflow,
- делает health‑checks,
- unpause/trigger `dag_data_predictions` и `dag_training`.

Опционально сбросить эксперименты MLflow при старте:
```
BOOTSTRAP_RESET_MLFLOW=true
```

---

## 7) Проверка сервинга (ручной smoke‑test)

### 7.1 Найти контейнер
```
docker ps --format '{{.Names}}' | grep mlflow-serve-
```

### 7.2 Health‑check
```
docker run --rm --network mlops_default curlimages/curl:8.5.0 -sS \
	http://<mlflow-serve-container>:5000/ping
```

### 7.3 Инференс
```
docker run --rm --network mlops_default curlimages/curl:8.5.0 -sS \
	-H 'Content-Type: application/json' \
	-d '{"dataframe_records":[{"feature_a":1,"feature_b":2}]}' \
	http://<mlflow-serve-container>:5000/invocations
```
Фактический порядок признаков берите из `data_contract/input_schema.json`.

---

## 8) Переменные окружения (.env)
Ключевые переменные:
- `MLFLOW_TRACKING_URI` — URL MLflow внутри сети.
- `MLFLOW_EXPERIMENT_NAME` — базовое имя эксперимента.
- `MLFLOW_MODEL_NAME` — базовое имя модели.
- `MLFLOW_SERVE_ALIASES` — aliases, которые отслеживает autoserve.
- `S3_ARTIFACT_BUCKET` / `MLFLOW_S3_ENDPOINT_URL` — MinIO.
- `AIRFLOW_WEB_PORT`, `MLFLOW_PORT`, `GRAFANA_PORT` и т.п. — внешние порты.

---

## 9) Основные адреса
- MLflow: http://localhost:${MLFLOW_PORT}
- Airflow: http://localhost:${AIRFLOW_WEB_PORT}
- MinIO Console: http://localhost:${MINIO_CONSOLE_PORT}
- Grafana: http://localhost:${GRAFANA_PORT}
- Prometheus: http://localhost:${PROMETHEUS_PORT}
- Loki: http://localhost:${LOKI_PORT}

---

## 10) Типичные проблемы и решения

### 10.1 Нет контейнера `mlflow-serve-*`
- Проверьте, что alias `Production` назначен модели.
- Посмотрите логи `mlflow-autoserve`.

### 10.2 /metrics возвращает 404
- Это ожидаемо для MLflow Serve.
- Используйте health‑checks через Blackbox.

### 10.3 Предупреждения о версиях зависимостей
- Возможны предупреждения из-за несовпадения `scikit-learn` или Python.
- В прод‑режиме это устраняется сборкой image на версию модели (dependencies «запечены»).

### 10.4 Git не добавляет positions.yaml
- Файл `monitoring/promtail/positions/positions.yaml` игнорируется в `.gitignore`.

---

## 11) Что считать успехом демо
- Все контейнеры `Up`.
- DAG‑и завершаются без ошибок.
- В MLflow есть модели и alias `Production`.
- `mlflow-serve-*` поднялись и видны в Grafana.
- Grafana показывает health‑статусы и логи без критических ошибок.

---

## 12) Краткий чеклист (5 минут)
1) `docker compose ps` → все сервисы `Up`.
2) Airflow → `dag_data_predictions` и `dag_training` в `success`.
3) MLflow → есть модель и alias `Production`.
4) `mlflow-autoserve` → в логах запуск `mlflow-serve-*`.
5) Grafana → `Service Health Detailed` показывает `mlflow-serve-*` в `UP`.

