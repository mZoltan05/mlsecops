FROM apache/airflow:2.8.1-python3.10

WORKDIR /airflow

COPY train_model_dag.py /airflow/dags/train_model_dag.py

EXPOSE 8080

CMD ["airflow", "standalone"]