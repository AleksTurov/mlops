# Архитектура и операционная модель (RU)

Этот документ рассказывает ту же историю, что и основной README, но со стороны архитектуры: из чего состоит стек, почему alias-driven deployment меняет модель эксплуатации, и что реально происходит после `make demo`.

Связанные документы:
- [README.md](../README.md)
- [SIMPLE_DIAGRAM.md](SIMPLE_DIAGRAM.md)
- [DEMO.md](DEMO.md)
- [CONFERENCE_SCRIPT.md](CONFERENCE_SCRIPT.md)

## 1) Главная идея

**Deployment is not a pipeline. Deployment is a label.**

В этой архитектуре выкладка модели не равна отдельному CI/CD-процессу.

Выкладка модели равна переключению alias в MLflow Registry.

Это значит, что мы можем:
- мгновенно выкладывать новую модель,
- делать rollback за секунды,
- не держать отдельный кастомный deployment orchestrator,
- использовать MLflow Registry как источник истины для serving.

Именно в этом и состоит core innovation проекта: promotion и rollback становятся операциями над metadata, а не отдельным инфраструктурным сценарием.

## 2) Что входит в стек

Стек поднимает полноценный локальный MLOps-контур:
- **MLflow** хранит experiments, registry моделей, alias и traces.
- **MinIO** хранит артефакты моделей.
- **Airflow** в этом demo готовит данные, запускает обучение и bootstrap-сценарий.
- **MLflow Autoserve** следит за alias и поднимает serving-контейнеры автоматически.
- **Prometheus + Grafana + Loki** дают health, метрики, логи и наблюдаемость.

При этом Airflow здесь не является обязательным требованием архитектуры. Модель можно обучить и из notebook, а затем зарегистрировать в MLflow Registry. Источником данных могут быть batch-таблицы, application DB или feature store.

Итоговый поток очень простой:

```mermaid
flowchart LR
    A[Notebook or Airflow train] --> B[MLflow model version]
    B --> C[Alias champion or challenger]
    C --> D[Autoserve reconcile]
    D --> E[Online or offline serving path]
    E --> F[Grafana plus Prometheus and Loki]
```

## 3) End-to-End Flow

1. Данные приходят из batch-источников, application DB или feature store.
2. Модели обучаются в Airflow или notebook.
3. Лучшая версия логируется и регистрируется в MLflow.
4. Alias в MLflow указывает на активную model version.
5. Autoserve пересоздает `mlflow-serve-*` контейнеры под новый target.
6. Grafana показывает состояние сервисов и endpoint'ов через Prometheus и Loki.
7. Bootstrap path записывает prediction traces в MLflow.

Результат: train, register, promote, serve и observe собраны в одном локально воспроизводимом стенде.

## 4) Почему эта модель работает

Главная сильная сторона не в количестве сервисов, а в том, что deployment становится дешевой, быстрой и обратимой операцией.

### Практические преимущества
- **Быстрый rollout**: новая версия появляется через alias, а не через отдельный deployment pipeline.
- **Мгновенный rollback**: достаточно вернуть alias на предыдущую версию.
- **Прозрачность**: registry, serving, health и traces видны в одном контуре.
- **Минимум кастомного кода**: serving построен на стандартном MLflow Serve.
- **Один воспроизводимый контур**: тот же стек подходит для demo, разработки и диагностики.
- **Гибкость обучения**: обучение может идти через Airflow, notebook или другой pipeline, если итоговая модель попадает в MLflow Registry.
- **Гибкость использования**: паттерн подходит и для online inference, и для offline scoring.
- **Понятная демонстрация на сцене**: train → alias → auto-deploy видно сразу и без дополнительных пояснений.

## 5) Что стартует автоматически

После `make demo` стенд автоматически поднимает:
- MinIO bucket для MLflow artifacts
- Airflow metadata и demo admin user
- `dag_data_predictions` и `dag_training`
- alias-driven autoserve для `champion` и `challenger`
- prediction path, который пишет явные MLflow traces
- Grafana, Prometheus, Loki, Promtail и Blackbox monitoring

Публичный репозиторий не использует Vault и работает в isolated Compose project `mlops-demo` с сетью `mlops-demo_default`.

## 6) Serving path

- каждый alias создает контейнер вида `mlflow-serve-<model>-<alias>`
- health endpoint: `GET /ping`
- inference endpoint: `POST /invocations`
- autoserve передает source experiment из model version run внутрь serving container
- для demo payload берется из `data_contract/sample_input.csv`, который логируется во время training

Основные entry points для запросов:
- [../test/test_integration_predictions.py](../test/test_integration_predictions.py)
- [../scripts/predict_request.py](../scripts/predict_request.py)
- [../scripts/print_model_input_schema.py](../scripts/print_model_input_schema.py)

## 7) Traditional vs This Approach

| Шаг | Традиционный подход | Этот проект |
|---|---|---|
| Deployment | CI/CD pipeline | Переключение alias |
| Rollback | Ручной redeploy | Мгновенный alias move |
| Serving | Кастомный API сервис | MLflow serve |
| Release target | Среда / environment | Alias в registry |
| Validation | Отдельный релизный процесс | `challenger` рядом с `champion` |

## 8) Компоненты

### MLflow
- хранит experiments,
- хранит registered models и versions,
- держит alias `champion` и `challenger`,
- отображает traces demo-запросов.

### MinIO
- хранит MLflow artifacts,
- содержит модель, зависимости и data contract.

### Airflow
- запускает `dag_data_predictions`,
- запускает `dag_training`,
- через `demo-bootstrap` поднимает demo в рабочее состояние.

Важно: для самой идеи alias-driven deployment Airflow не обязателен. Он просто выбран как orchestration layer в этой публичной demo-сборке.

### MLflow Autoserve
- читает registry,
- находит версии за alias,
- пересоздает `mlflow-serve-*` контейнеры,
- тем самым превращает alias в реальный deployment mechanism.

### Observability stack
- Prometheus и Blackbox проверяют доступность и хранят метрики,
- Grafana показывает состояние сервисов и alias, используя Prometheus и Loki как источники данных,
- Loki хранит логи,
- MLflow показывает traces demo-запросов.

Наблюдаемость здесь намеренно расположена рядом с механизмом выкладки: после изменения alias тот же контур сразу показывает health, логи и trace evidence.

## 9) Почему это хорошо работает на demo

Сценарий предельно простой:

1. **Step 1: train model**
2. **Step 2: assign alias**
3. **Step 3: watch auto-deploy**

Практически это выглядит так:
- в Airflow или notebook появляется успешный training run,
- в MLflow появляется новая model version,
- alias `champion` или `challenger` указывает на нужную версию,
- autoserve автоматически пересоздает serving-контейнер,
- Grafana показывает, что endpoint жив,
- MLflow показывает traces demo-запросов.

Этот сценарий работает убедительно, потому что deployment виден как операция над metadata, а не как отдельный инфраструктурный ceremony.

## 10) Где смотреть результат

### MLflow
- UI: `http://localhost:15000`
- experiment: `iris-classification_iris`
- вкладка `Models`: версии и alias
- вкладка `Traces`: demo-traces после bootstrap/test run

### Airflow
- UI: `http://localhost:18885`
- проверить `dag_data_predictions` и `dag_training`

### Grafana
- UI: `http://localhost:13000`
- открыть `MLOps Overview`
- открыть `Service Health Detailed`
- открыть `MLflow Serving`

### Serve контейнеры
- `mlflow-serve-iris_classifier_iris-champion`
- `mlflow-serve-iris_classifier_iris-challenger`

## 11) Что важно понимать про traces

Обычные `POST /invocations` в MLflow Serve не дали traces автоматически в этом стенде.

Поэтому demo path делает это явно:
- integration test отправляет реальный prediction request,
- после успешного ответа записывает trace через MLflow trace API,
- trace попадает в experiment `iris-classification_iris`.

Это делает demo предсказуемой: traces появляются всегда, без зависимости от скрытого поведения serve runtime.

## 12) Как быстро перепроверить demo

После `make demo` используйте:

```bash
make verify
```

Если нужен только повтор tested champion request path:

```bash
RUN_INTEGRATION_TESTS=1 .venv/bin/python -m pytest -q test/test_integration_predictions.py -k champion
```

## 13) Почему это сильный public demo

- Не нужен Vault.
- Не нужен внешний deployment pipeline.
- Не нужен отдельный custom serving layer.
- Все поднимается локально одной командой.
- Архитектура понятна и для инженеров, и для менеджмента.
- Подход не привязан только к online serving: его можно использовать и для offline scoring сценариев.

Если формулировать коротко:

**Мы превращаем deployment модели из тяжелой DevOps-операции в дешевое переключение alias внутри MLflow Registry.**

## 14) Что читать дальше

- [README.md](../README.md) для quick start и project positioning.
- [DEMO.md](DEMO.md) для запуска и проверки стенда.
- [CONFERENCE_SCRIPT.md](CONFERENCE_SCRIPT.md) для короткого выступления или live demo.
- [SCRIPTS.md](SCRIPTS.md) для справки по утилитам и DAG.

