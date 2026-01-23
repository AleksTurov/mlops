FROM python:3.10-slim

# Устанавливаем системные зависимости для psycopg2 и работы с s3
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# По умолчанию запускаем MLflow
CMD mlflow server \
    --backend-store-uri ${POSTGRES_URI} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000