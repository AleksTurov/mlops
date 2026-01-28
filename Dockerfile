FROM python:3.12-slim

# Install system dependencies for psycopg2 and S3 support
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run MLflow tracking server
CMD mlflow server \
    --backend-store-uri ${POSTGRES_URI} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000