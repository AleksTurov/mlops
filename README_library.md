# mlops-toolkit

Small helper library for data scientists to accelerate MLflow workflows and production testing.

Features
- MLflow helpers: alias management, artifact upload/download

Install (editable for local dev)
```bash
pip install -e .
```

Quick examples

Set MLflow alias
```bash
mlops-toolkit alias set --model-name my_model --version 12 --alias Production
```

Upload artifact
```bash
export MLFLOW_TRACKING_URI=http://localhost:${MLFLOW_PORT}
mlops-toolkit artifact upload --run-id <RUN_ID> --path ./model.pkl
```

 
