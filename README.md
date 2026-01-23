# 🚀 MLflow Tracking Server — Dockerized Setup

Контейнерная сборка MLflow (версия 3.5.0) с PostgreSQL как metadata store и MinIO как S3‑совместимым хранилищем артефактов. Сборка ориентирована на внутреннее развёртывание для разработки, тестирования и небольших окружений.

---

## 🔎 Структура репозитория

/data/aturov/mlflow/
- docker-compose.yml        — оркестрация (MLflow, Postgres, MinIO)
- Dockerfile                — (опционально) кастомный образ MLflow
- requirements.txt          — зависимости для локальной разработки
- .env.example              — пример конфигурации (без секретов)
- pgdata/                   — том Postgres (локально)
- minio_data/               — том MinIO (локально)
- README.md                 — документация (этот файл)
- test/test_mlflow.py       — пример smoke‑теста

> Примечание: реальный файл `.env` с паролями и ключами не должен храниться в репозитории.

---

## ⚙️ Быстрый старт
1️⃣ Перед запуском клиента экспортируйте S3‑переменные (локально, не в репо):

```bash
cd /data/aturov/mlflow 
# создать venv (если его нет)
python3 -m venv venv
# активировать
source venv/bin/activate
# установить зависимости (если есть requirements.txt)
pip install --upgrade pip
pip install -r requirements.txt
```
2️⃣ Start all services  
docker compose --env-file .env up -d --build  
  
3️⃣ Verify running containers  
docker ps  
You should see:  
  
mlflow_postgres  
mlflow_minio  
mlflow_server  

🌐 Accessing the Services    
Service	URL	Notes    
MLflow UI	http://10.16.230.222:5000    
	Main MLflow interface    
MinIO Console	http://10.16.230.222:9023      
MinIO S3 API	http://10.16.230.222:9022    
  Need creating a bucket named `mlflow` for MLflow artifacts storage    
PostgreSQL	10.16.230.222:6432	Accessible with pgAdmin or psql      
(Adjust ports as needed based on your `.env` configuration.)    
---

## 🧪 Smoke test (пример)
  
Файл: test/test_mlflow.py — пример логирования параметров, метрик и артефакта:

```python
import mlflow, tempfile, json, time, os

mlflow.set_tracking_uri("http://<MLFLOW_HOST>:5000")
mlflow.set_experiment("scoring-features")

with mlflow.start_run(run_name="smoke-test"):
    mlflow.log_param("p", 123)
    mlflow.log_metric("m", 0.42)
    with tempfile.TemporaryDirectory() as d:
        fpath = os.path.join(d, "sample.json")
        json.dump({"ok": True, "ts": time.time()}, open(fpath, "w"))
        mlflow.log_artifact(fpath)

print("✅ Run complete")
```
---

## 🧰 Полезные команды

- Просмотр логов:    
  docker compose --env-file .env logs -f    
- Остановить и удалить (контейнеры + тома):    
  docker compose down   
- Проверить состояние контейнеров:  
  docker ps -a  
- Очистка Docker:   
  docker system prune -af --volumes  

---



## 🔮 MLOps Roadmap

- [x] Deploy MLflow Tracking Server (Postgres + MinIO)
- [ ] Integrate with Airflow for model training pipelines
- [ ] Add model serving (FastAPI + MLflow Registry)
- [ ] Add monitoring (Prometheus + Grafana)
- [ ] Secure with HTTPS and authentication (Nginx)
- [ ] Automate backups and versioning (Postgres + MinIO)

## 🧑‍💻 Автор
Alexey Turov — Data Scientist @ Beeline Kyrgyzstan