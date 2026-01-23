import mlflow, tempfile, json, time, os
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path('/data/aturov/scoring/.env')
load_dotenv(dotenv_path=env_path)

mlflow.set_tracking_uri("http://10.16.230.222:5000")


mlflow.set_experiment("scoring")
with mlflow.start_run(run_name="smoke-test"):
    mlflow.log_param("p", 123)
    mlflow.log_metric("m", 0.42)
    with tempfile.TemporaryDirectory() as d:
        fpath = os.path.join(d, "sample.json")
        json.dump({"ok": True, "ts": time.time()}, open(fpath, "w"))
        mlflow.log_artifact(fpath)
print("done")
