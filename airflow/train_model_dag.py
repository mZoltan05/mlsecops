from datetime import timedelta

from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.utils.dates import days_ago

# Define the DAG
with DAG(
    dag_id="train_model_dag",
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="Trigger /train endpoint to start model training",
    schedule_interval=None,  # Run manually or triggered externally
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    # Define HTTP training task
    trigger_training = SimpleHttpOperator(
        task_id="trigger_model_training",
        http_conn_id=None,
        endpoint="http://172.31.0.2:8000/train",
        method="POST",
        headers={"Content-Type": "application/json"},
        data="""
        {
            "data_root": "/app/data",
            "out_dir": "/app/checkpoints",
            "device": "cpu",
            "epochs": 1,
            "batch_size": 16,
            "img_size": 224,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "num_workers": 1,
            "seed": 42,
            "pretrained": true
        }
        """,
        log_response=True,
    )

    trigger_training
