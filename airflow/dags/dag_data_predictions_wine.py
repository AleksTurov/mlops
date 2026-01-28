from datetime import timedelta
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.append("/opt/airflow/src")

from ml.data import load_sklearn_dataset


def load_data_task() -> None:
    load_sklearn_dataset("wine")


with DAG(
    dag_id="dag_data_predictions_wine",
    description="Manual data load for wine experiment",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:
    PythonOperator(task_id="load_wine_data", python_callable=load_data_task)
