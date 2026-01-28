from datetime import timedelta
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.append("/opt/airflow/src")

from ml.inference import run_inference, run_shadow_inference


def inference_task() -> None:
    run_inference()


def shadow_task() -> None:
    run_shadow_inference()


with DAG(
    dag_id="dag_inference_wine",
    description="Manual inference for wine experiment",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:
    PythonOperator(task_id="run_inference", python_callable=inference_task)
    PythonOperator(task_id="run_shadow_inference", python_callable=shadow_task)
