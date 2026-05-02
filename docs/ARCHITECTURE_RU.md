# Архитектура и поток данных (RU)

Ниже — подробное практическое описание архитектуры, потоков данных и эксплуатации. Цель документа — не только перечислить сервисы, но и объяснить **что именно происходит по шагам**, какой компонент за что отвечает, какие данные где лежат и как быстро проверить, что контур работает корректно.

---

## 1) Общая идея
Проект реализует минимальный, но полноценный MLOps‑контур:
- MLflow хранит эксперименты, метрики и registry моделей.
- MinIO хранит артефакты моделей (S3‑совместимое хранилище).
- Airflow выполняет пакетные задачи (данные, обучение, batch‑инференс, мониторинг).
- Автоматический онлайн‑сервинг строится на **MLflow Serve** контейнерах, которые поднимаются по alias (`champion`, `challenger`).
- Наблюдаемость обеспечивается Prometheus + Grafana + Loki/Promtail + Blackbox.

Практическая цель: получить прозрачную цепочку **данные → обучение → версия модели → alias → онлайн/батч инференс → мониторинг и диагностика**.

---

## 2) Компоненты и роли

### 2.1 MLflow и хранилища
- **MLflow Tracking Server**
	- Принимает логи runs (params, metrics, tags, artifacts).
	- Хранит Model Registry (имя модели, версии, alias).
	- Служит источником правды для autoserve: какая версия сейчас стоит за alias.
- **PostgreSQL (`mlflow-db`)**
	- Хранит метаданные MLflow (эксперименты, runs, версии моделей, alias‑привязки).
	- Не хранит тяжелые бинарные артефакты модели.
- **MinIO**
	- Хранит артефакты run/model (MLmodel, сериализованная модель, schema/metrics файлы и пр.).
	- Используется MLflow как artifact store.

### 2.2 Airflow и операционный контур данных
- **Airflow Scheduler + Webserver**
	- Планирует и исполняет DAG‑и по расписанию/триггеру.
	- Даёт UI для ручного запуска/повтора задач.
- **PostgreSQL приложения (внешний, не локальный сервис compose)**
	- Хранит операционные данные проекта: исходные данные, промежуточные таблицы, результаты batch‑предсказаний.
	- Используется DAG‑ами как рабочее хранилище и передаётся через Vault‑переменные (`APP_DB_*`/`APP_DB_URL`).

### 2.3 Сервинг моделей
- **MLflow Autoserve (`mlflow-autoserve`)**
	- Периодически читает registry + alias из MLflow.
	- Для каждой связки `model@alias` определяет target version.
	- Поднимает/обновляет соответствующий `mlflow models serve` контейнер.
	- При смене alias (например, challenger→новая версия) пересоздает serving‑контейнер.
- **MLflow Serve контейнеры**
	- Каждый контейнер отвечает за один alias конкретной модели.
	- Пример логического имени: `model_name@challenger` или `model_name@champion`.
	- По умолчанию контейнеры маркируются проектами: `champion -> models_champion`, `challenger -> models_challenger`.
	- Имеют стандартный API: `/ping` и `/invocations`.

### 2.4 Наблюдаемость
- **Prometheus**
	- Снимает технические метрики доступности/latency с Blackbox exporter и инфраструктурных endpoints.
- **Blackbox exporter**
	- Выполняет HTTP‑пробы (`/ping`) по целям, включая alias‑сервинг контейнеры.
- **Grafana**
	- Визуализирует статус сервисов/alias и latency.
	- Используется для оперативной диагностики и алертов.
- **Loki + Promtail**
	- Promtail читает Docker‑логи контейнеров и отправляет в Loki.
	- Grafana Explore позволяет делать выборки логов по labels (`service`, `container`, `project`).

---

## 3) Что происходит после `docker compose up` (runtime bootstrap)

Ниже — реальная последовательность в рантайме:

1. Поднимаются базовые инфраструктурные сервисы: БД, MinIO, MLflow, Airflow, мониторинг.
2. MLflow подключается к `mlflow-db` и к MinIO как artifact store.
3. Airflow поднимает scheduler/webserver и становится готов к DAG‑ам.
4. Autoserve стартует и начинает цикл синхронизации с MLflow Registry.
5. Если в registry есть модели с alias из `MLFLOW_SERVE_ALIASES`, autoserve поднимает serving‑контейнеры.
6. Blackbox начинает health‑пробы `GET /ping` по всем целям.
7. Prometheus собирает результаты проб, Grafana показывает статусы, Promtail отправляет логи в Loki.

Итог: если alias назначены корректно, через короткое время в Grafana видно состояние как базовых сервисов, так и alias‑сервинга.

---

## 4) Потоки данных (end‑to‑end)

### 4.1 Подготовка данных
**DAG:** `dag_data_predictions`
- Читает/генерирует входные данные (например, iris).
- Пишет таблицы во внешнюю application DB.
- Фиксирует операционный baseline для последующих DAG‑ов (обучение/инференс).

Что важно проверять:
- DAG завершился со статусом `success`.
- Целевые таблицы во внешней application DB созданы и содержат записи.

### 4.2 Обучение и регистрация модели
**DAG:** `dag_training`
- Вызывает логику обучения (`ml.training.train_candidate()` или аналог).
- Логирует в MLflow: params, metrics, tags, artifacts.
- Регистрирует модель в Model Registry и создаёт новую версию.
- Назначает alias (обычно `champion`/`challenger`) согласно правилам проекта.

Что физически создаётся:
- Запись run в MLflow DB.
- Артефакты run/model в MinIO bucket.
- Новая версия модели в Registry.
- Привязка alias → version.

### 4.3 Онлайн‑сервинг по alias
**Сервис:** `mlflow-autoserve`
- Читает alias‑состояние в registry.
- Для каждого alias вычисляет нужную версию модели.
- Поднимает `mlflow models serve` c URI вида `models:/<model_name>@<alias>` (или эквивалентом по версии).
- Если alias переведен на другую версию — перезапускает контейнер на новую модель.

Ключевой принцип: **сервинг следует alias, а не «последнему run»**.

### 4.4 Batch‑инференс
**DAG:** `dag_inference`
- Читает батч входных данных из внешней application DB.
- Загружает модель через MLflow (обычно через alias/версию).
- Считает предсказания и пишет результат во внешнюю application DB.

Назначение batch‑контура:
- Регулярные пакетные расчеты для витрин/отчетов.
- Сравнение с онлайн‑сценарием и контроль стабильности.

### 4.5 Мониторинг качества
**DAG:** `dag_model_monitoring`
- Сравнивает candidate vs production (метрики качества и/или стабильности).
- Логирует результаты сравнения в MLflow.
- Даёт основу для решения «продвигать ли кандидат в champion».

---

## 5) Механика MLflow Autoserve (детально)

### 5.1 Цикл синхронизации autoserve
На каждой итерации autoserve делает:
1) Запрос списка моделей в registry.
2) Проверку alias из `MLFLOW_SERVE_ALIASES`.
3) Разрешение alias → version.
4) Сверку желаемого состояния с текущими контейнерами.
5) Создание/обновление/остановку контейнеров для достижения нужного состояния.

### 5.2 Build per model version image
В режиме «build per version»:
- Для версии модели создаётся image `mlflow-model-<model>-v<version>`.
- Сборка через MLflow фиксирует зависимости внутри image.
- Это снижает риск несовместимостей (python/sklearn) между версиями модели.

### 5.3 Метки контейнеров
Autoserve пишет Docker labels, например:
- `mlflow_model`
- `mlflow_alias`
- `mlflow_version`
- `mlflow_image`
- `com.docker.compose.project` (`models_champion`/`models_challenger`)

Эти labels нужны для:
- корректного reconcile‑цикла (понимать, что уже запущено),
- удобной фильтрации в мониторинге/логах.

### 5.4 API serving‑контейнера
- `GET /ping` — health endpoint (используется blackbox‑пробами).
- `POST /invocations` — инференс в формате MLflow scoring.

### 5.5 Контракт входных данных
Рекомендуется хранить вместе с моделью:
- `data_contract/input_schema.json` — обязательные признаки и типы.
- `data_contract/sample_input.csv` — пример валидного payload.
- `metrics/validation_metrics.json` — метрики качества на валидации.

Практический смысл: контракт уменьшает количество ошибок вида «не те поля/порядок/типы» при интеграции клиентов.

---

## 6) Наблюдаемость и что смотреть в первую очередь

### 6.1 Service Health (Grafana)
Дашборд показывает:
- базовые сервисы (`MLflow`, `Airflow`, `MinIO`, `Prometheus`, `Loki`, `Grafana`),
- alias‑сервинг (`model@alias`) как отдельные цели.

Если тут `DOWN`, это обычно инфраструктурная проблема (контейнер не поднят, сеть, endpoint).

### 6.2 MLflow Serving dashboard
Полезные сигналы:
- `probe_success` — доступность alias endpoint.
- `probe_duration_seconds` — latency health‑ответа.

Если `probe_success=0`, проверяем цепочку: alias в registry → контейнер autoserve → `/ping`.

### 6.3 Логи через Loki
В Grafana Explore обычно хватает фильтров:
- `project=models_champion` или `project=models_challenger` (для serving‑контейнеров)
- `project=mlops` (для базовой инфраструктуры)
- `service=mlflow-autoserve` или `service=airflow`.

Стандартный путь диагностики:
1) Проверить ошибку на дашборде,
2) Открыть логи соответствующего сервиса,
3) Убедиться, что причина локализована (неверный alias, недоступный MLflow, ошибка зависимостей и т.д.).

### 6.4 Почему у MLflow Serve нет `/metrics`
Это нормальное поведение: у стандартного MLflow Serve нет native Prometheus‑метрик.
Поэтому базовый мониторинг строится на:
- blackbox health‑пробах,
- логах,
- косвенных SLI (доступность + latency ping).

Для полноценных API‑метрик (RPS, p95, 5xx) обычно добавляют gateway/sidecar.

---

## 7) Первый запуск и быстрый сценарий проверки

```bash
set -a
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/mlops
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/grafana
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/minio
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/mlflow
source /data/aturov/vault/scripts/export-env.sh kv/data/dev/airflow
set +a
docker compose up -d --build
```

Если используется `demo-bootstrap`, он обычно:
- ждёт готовности MLflow/Airflow,
- выполняет базовые health‑проверки,
- триггерит/размораживает DAG‑и подготовки и обучения.

Дополнительно (опционально):
```bash
BOOTSTRAP_RESET_MLFLOW=true
```
Использовать только для «чистого» демо‑прогона, когда допустим сброс состояния экспериментов.

---

## 8) Ручной smoke‑test сервинга

### 8.1 Найти serving‑контейнеры
```bash
docker ps --format '{{.Names}}' | grep mlflow-serve-
```

### 8.2 Проверить health
```bash
docker run --rm --network mlops_default curlimages/curl:8.5.0 -sS \
	http://<mlflow-serve-container>:5000/ping
```
Ожидаем успешный ответ и отсутствие таймаута.

### 8.3 Проверить инференс
```bash
docker run --rm --network mlops_default curlimages/curl:8.5.0 -sS \
	-H 'Content-Type: application/json' \
	-d '{"dataframe_records":[{"feature_a":1,"feature_b":2}]}' \
	http://<mlflow-serve-container>:5000/invocations
```

Важно: структура payload должна соответствовать `data_contract/input_schema.json`.

---

## 9) Переменные окружения (Vault export): что на что влияет

Ключевые переменные:
- `MLFLOW_TRACKING_URI` — куда клиенты/сервисы отправляют MLflow API запросы.
- `MLFLOW_EXPERIMENT_NAME` — имя эксперимента по умолчанию для логирования run.
- `MLFLOW_MODEL_NAME` — базовое имя модели в registry.
- `MLFLOW_SERVE_ALIASES` — какие alias autoserve обязан обслуживать (например, `champion,challenger`).
- `MLFLOW_SERVE_ALIAS_PROJECTS` — соответствие alias→project (по умолчанию `champion=models_champion,challenger=models_challenger`).
- `MLFLOW_SERVE_PROJECT_CHAMPION` / `MLFLOW_SERVE_PROJECT_CHALLENGER` — точечные override для конкретных alias.
- `S3_ARTIFACT_BUCKET` / `MLFLOW_S3_ENDPOINT_URL` — куда MLflow пишет артефакты.
- `AIRFLOW_WEB_PORT`, `MLFLOW_PORT`, `GRAFANA_PORT` и др. — внешние порты для UI/API.

Операционное правило: если в Vault меняются alias/URI/порты, заново экспортируйте переменные перед `docker compose up` и затем проверяйте `docker compose config`.

---

## 10) Основные адреса
- MLflow: http://localhost:${MLFLOW_PORT}
- Airflow: http://localhost:${AIRFLOW_WEB_PORT}
- MinIO Console: http://localhost:${MINIO_CONSOLE_PORT}
- Grafana: http://localhost:${GRAFANA_PORT}
- Prometheus: http://localhost:${PROMETHEUS_PORT}
- Loki: http://localhost:${LOKI_PORT}

---

## 11) Типичные проблемы и как локализовать

### 11.1 Нет контейнера `mlflow-serve-*`
Проверка по шагам:
1) Есть ли модель и alias в MLflow Registry.
2) Входит ли alias в `MLFLOW_SERVE_ALIASES`.
3) Есть ли ошибки в логах `mlflow-autoserve` (доступ к MLflow, сборка image, запуск контейнера).

### 11.2 `probe_success=0` в Grafana
Обычно причина в одном из трёх мест:
- контейнер сервинга не запущен,
- endpoint недоступен по сети,
- контейнер стартовал, но упал при инициализации модели.

### 11.3 `/metrics` возвращает 404
Это ожидаемо для MLflow Serve. Используйте blackbox + логи.

### 11.4 Предупреждения о версиях зависимостей
Возможны на окружениях с разными версиями Python/библиотек.
Стабильный путь — build per model version image.

### 11.5 Git не добавляет `positions.yaml`
`monitoring/promtail/positions/positions.yaml` — runtime state файл, он игнорируется в `.gitignore`.

---

## 12) Что считать успешным состоянием платформы
- Все критичные контейнеры в статусе `Up`.
- DAG‑и подготовки/обучения выполняются в `success`.
- В MLflow видны версии модели и alias (как минимум `champion`, опционально `challenger`).
- Сервинг‑контейнеры `mlflow-serve-*` подняты по нужным alias.
- Grafana показывает `UP` по сервисам и alias‑целям.
- В логах нет постоянных критических ошибок (допускаются кратковременные стартовые предупреждения).

---

## 13) Краткий чеклист (5 минут)
1) `docker compose ps` → базовые сервисы `Up`.
2) Airflow → `dag_data_predictions` и `dag_training` в `success`.
3) MLflow → есть версия модели и назначенный alias.
4) `mlflow-autoserve` → в логах есть reconcile и запуск `mlflow-serve-*`.
5) Grafana → `Service Health Detailed` показывает alias‑сервинг в `UP`.
6) Smoke‑test `POST /invocations` возвращает валидный ответ.

