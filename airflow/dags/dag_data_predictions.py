from datetime import timedelta
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.append("/opt/airflow/src")

from ml.data import load_sklearn_dataset


def load_data_task() -> None:
    load_sklearn_dataset("iris")


with DAG(
    dag_id="dag_data_predictions",
    description="Daily data refresh",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:
    PythonOperator(task_id="load_iris_data", python_callable=load_data_task)
