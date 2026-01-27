# Scripts and Purpose (EN)

### scripts/predict_request.py
- **Why**: send a request to model‑server and validate response.
- **What**: reads JSON payload and calls `/predict`.
- **When**: smoke test and demo.

## Airflow DAGs (scheduled scripts)

### airflow/dags/dag_data_predictions.py
- **Why**: prepare data for training/inference.
- **What**: loads a sklearn dataset (iris) into app-db.

### airflow/dags/dag_training.py
- **Why**: regular model training and registration.
- **What**: runs `ml.training.train_candidate()` weekly.

### airflow/dags/dag_inference.py
- **Why**: regular inference pipeline.
- **What**: runs `run_inference()` and `run_shadow_inference()` hourly.

### airflow/dags/dag_model_monitoring.py
- **Why**: compare candidate vs production quality.
- **What**: runs `ml.training.evaluate_models()` daily.
