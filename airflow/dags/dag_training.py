from datetime import timedelta
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

sys.path.append("/opt/airflow/src")

from ml.training import train_candidate


def train_task() -> None:
    train_candidate()


with DAG(
    dag_id="dag_training",
    description="Weekly model training and registration",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
) as dag:
    PythonOperator(task_id="train_models", python_callable=train_task)
