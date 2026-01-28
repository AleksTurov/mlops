# Архитектура и поток данных (RU)

Эта версия — минимальный, но наглядный шаблон MLOps: любой может запустить и сразу увидеть статус сервисов, ошибки, логи и метрики сервинга.

## 1) Роли сервисов (минимальный набор)
- **MLflow** — реестр моделей и трекинг экспериментов.
- **PostgreSQL (mlflow-db)** — метаданные MLflow.
- **MinIO** — артефакты моделей (S3‑совместимо).
- **PostgreSQL (app-db)** — данные и предсказания.
- **Airflow** — пайплайны: загрузка данных, обучение, инференс.
- **MLflow Autoserve** — сервис‑наблюдатель: при появлении alias запускает отдельный MLflow Serve.
- **Prometheus** — сбор метрик.
- **Grafana** — дашборды и алерты.
- **Loki/Promtail** — сбор и поиск логов.
- **Blackbox exporter** — проверки health‑endpoint (жив/не жив).

## 2) Поток данных (подробно)
1. **Загрузка данных** → DAG `dag_data_predictions` пишет данные в **app-db**.
2. **Обучение** → DAG `dag_training` запускает `ml.training.train_candidate()`.
	- Метрики/параметры → **MLflow**.
	- Артефакты моделей → **MinIO**.
3. **Алиас модели** → DS вручную назначает алиас (например, `Production`) в MLflow UI.
4. **Сервинг** → `mlflow-autoserve` обнаруживает alias и поднимает отдельный `mlflow models serve`.
	- При смене alias контейнер пересоздаётся на новую версию.
5. **Инференс** → DAG `dag_inference` записывает предсказания в **app-db**.
6. **Наблюдаемость**:
	- Метрики сервинга → Prometheus/Grafana.
	- Логи → Loki/Grafana.
	- Health‑checks → Blackbox/Grafana.

## 3) Наблюдаемость (жив/не жив, ошибки, логи)
**Статус сервисов**
- Проверки HTTP health‑endpoint через Blackbox exporter.
- В Grafana видно, кто «жив» и какие коды ответа возвращаются.

**Ошибки**
- Алёрты по `probe_success == 0` (сервис недоступен).

**MLflow Serve**
- Проверки `/ping` для всех поднятых сервисов через Blackbox.
- Логи контейнеров доступны в Loki.
- В Grafana (дашборд **Service Health Detailed**) видны поля `model`, `alias`, `instance`.

**Логи**
- Promtail читает Docker‑логи и отправляет в Loki.
- В Grafana можно фильтровать по `service` и `container`.

**Дашборды**
- Все дашборды лежат в `monitoring/grafana/dashboards-min`.
- Основные панели: **MLOps Overview**, **Service Health**.

## 4) Логика mlflow-autoserve
- Периодически читает aliases из MLflow Registry.
- На каждый alias запускает `mlflow models serve` контейнер.
- Если alias указывает на новую версию — контейнер пересоздаётся.

## 5) Базы данных и сеть
По умолчанию используются **раздельные Postgres** (`mlflow-db`, `airflow-db`, `app-db`).
Это проще для изоляции и обслуживания.

**Сеть**
- Docker Compose создаёт общий network, сервисы общаются по DNS‑именам контейнеров.

## 6) Основные адреса
- MLflow: http://localhost:${MLFLOW_PORT}
- Airflow: http://localhost:${AIRFLOW_WEB_PORT}
- MinIO Console: http://localhost:${MINIO_CONSOLE_PORT}
- Grafana: http://localhost:${GRAFANA_PORT}
- Prometheus: http://localhost:${PROMETHEUS_PORT}
- Loki: http://localhost:${LOKI_PORT}

## 7) Демо‑запуск (пошагово)
Ниже — «проверочный сценарий», чтобы убедиться, что весь конвейер работает от данных до сервинга и мониторинга.

### Шаг 1. Поднять сервисы
```bash
cp env.dev.example .env
docker compose --env-file .env up -d --build
```
**Что увидеть:** все контейнеры в статусе `Up` (команда `docker compose ps`).

### Шаг 2. Загрузить данные
В Airflow UI запустите DAG `dag_data_predictions`.

**Что делает:** загружает датасет `iris` в `app-db`.
**Что увидеть:**
- В Airflow — `load_iris_data` в `success`.
- В логах DAG — сообщения о загрузке данных.

### Шаг 3. Обучить и залогировать модели
Запустите DAG `dag_training`.

**Что делает:** вызывает `ml.training.train_candidate()` и логирует метрики/артефакты в MLflow.
**Что увидеть:**
- В Airflow — `train_models` в `success`.
- В MLflow UI — новые эксперименты/запуски и зарегистрированные модели.

### Шаг 4. Продвинуть лучшую модель
В MLflow UI установите алиас `Production` на лучший run.

**Что увидеть:**
- В MLflow у модели появился алиас `Production`.
- В логах `mlflow-autoserve` сообщение о запуске `mlflow models serve` для нового alias.

### Шаг 5. Инференс в пайплайне
Запустите DAG `dag_inference`.

**Что делает:** запускает `run_inference()` и `run_shadow_inference()`.
**Что увидеть:**
- В Airflow — обе задачи в `success` (или `up_for_retry`, если включён ретрай на ошибках).
- В `app-db` — новые предсказания.
- В Grafana — сервисы `mlflow-serve-*` появляются в `Service Health` (probe_success).

### Шаг 6. Проверка сервинга
**Что проверить:**
- В Airflow DAG‑ах нет ошибок.
- В Grafana `Service Health` — новые `mlflow-serve-*` в `UP`.
- В `Service Health Detailed` отображаются `model`, `alias`, `instance`.

## 8) Какие скрипты и DAG‑и используются
**DAG: dag_data_predictions**
- **Зачем:** подготовка данных для обучения/инференса.
- **Куда смотреть:** Airflow (успех задачи), `app-db` (таблицы с `iris`).

**DAG: dag_training**
- **Зачем:** регулярное обучение и регистрация моделей.
- **Куда смотреть:** MLflow UI (метрики, модели), логи Airflow.

**DAG: dag_inference**
- **Зачем:** периодический инференс и запись предсказаний.
- **Куда смотреть:** Airflow + `app-db`, Grafana (probe_success для mlflow‑serve).

**DAG: dag_model_monitoring**
- **Зачем:** сравнение candidate vs production.
- **Куда смотреть:** логи задачи и MLflow (метрики сравнения).

## 9) Куда смотреть и что должно быть видно
**Airflow UI**
- Статусы задач `success`, время выполнения, логи задач.

**MLflow UI**
- Новые эксперименты/запуски, зарегистрированные модели.
- Алиасы (`Production`) назначены на нужную версию.

**MLflow Autoserve**
- В логах сервиса появляются записи о запуске контейнеров `mlflow-serve-*`.

**Grafana**
- Дашборды: **MLOps Overview**, **Service Health**.
- Новые сервисы отображаются по `probe_success`.

**Как обращаться к MLflow Serve**
- Найти контейнер: `docker ps --format '{{.Names}}' | grep mlflow-serve-`
- Запрос: `POST http://<container>:5000/invocations` (внутри Docker сети).
- Контракт входа: MLflow UI → Artifacts → `data_contract/input_schema.json`.

**Prometheus**
- Страница `/targets` — все таргеты в `UP`.

**Loki (через Grafana Explore)**
- Логи по лейблам `service`, `container`, `project=mlops`.

## 10) Что считать «успехом» демо
- Все контейнеры `Up`.
- DAG‑и завершаются без ошибок.
- В MLflow есть запуски и модель с алиасом `Production`.
- `mlflow-serve-*` поднялись и видны в `Service Health`.
- Grafana показывает метрики и логи без критических ошибок.

