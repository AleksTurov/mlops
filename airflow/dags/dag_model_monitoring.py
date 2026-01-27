from datetime import timedelta
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.append("/opt/airflow/src")

from ml.training import evaluate_models


def monitor_task() -> None:
    evaluate_models()


with DAG(
    dag_id="dag_model_monitoring",
    description="Monitor candidate vs production metrics",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:
    PythonOperator(task_id="evaluate_models", python_callable=monitor_task)
