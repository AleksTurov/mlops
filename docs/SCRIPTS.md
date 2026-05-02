# Scripts and Purpose (EN)

### scripts/predict_request.py
- **Why**: send a request to a model endpoint and validate response.
- **What**: reads MLflow scoring payload and calls `/invocations`.
- **When**: smoke test and demo (if a model endpoint is exposed).

### scripts/print_model_input_schema.py
- **Why**: inspect what a registered MLflow model version expects on input.
- **What**: resolves a model alias and prints any available schema metadata from `data_contract/input_schema.json`, `model/MLmodel`, `model/serving_input_example.json`, and `model/input_example.json`.
- **When**: before building a scoring payload or when validating a newly promoted champion/challenger model.

## Airflow DAGs (scheduled scripts)

### airflow/dags/dag_data_predictions.py
- **Why**: prepare data for training/inference.
- **What**: loads a sklearn dataset (iris) into the external application DB.

### airflow/dags/dag_training.py
- **Why**: regular model training and registration.
- **What**: runs `ml.training.train_candidate()` weekly.

### airflow/dags/dag_inference.py
- **Why**: regular inference pipeline.
- **What**: runs `run_inference()` and `run_shadow_inference()` hourly.

### airflow/dags/dag_model_monitoring.py
- **Why**: compare candidate vs production quality.
- **What**: runs `ml.training.evaluate_models()` daily.

## Monitoring components
- **Loki/Promtail**: log aggregation for Grafana.
- **MLflow Autoserve**: watches MLflow aliases and starts `mlflow models serve` containers.
