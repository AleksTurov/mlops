FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run MLflow tracking server
CMD mlflow server \
    --backend-store-uri ${POSTGRES_URI} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --allowed-hosts "${MLFLOW_SERVER_ALLOWED_HOSTS:-*}" \
    --host 0.0.0.0 \
    --port 5000