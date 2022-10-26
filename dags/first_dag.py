from datetime import datetime


from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def process_datetime(ti):
    dt = ti.xcom_pull(task_ids=['get_datetime'])
    if not dt:
        raise Exception('No datetime value.')
    dt = str(dt).split()

    return {
        'year': dt[3],
        'month': dt[2],
        'day': dt[1],
        'time': dt[4],
        'week_day': dt[0] 
    }


with DAG(
    dag_id='first_airflow_dag',
    schedule_interval="* * * * *",
    start_date=datetime(year=2022, month=2, day=1),
    catchup=False
) as dag:
    # Get current date
    task_get_datetime = BashOperator(
        task_id='get_datetime',
        bash_command='date'
    )

    # Process current date
    task_process_datetime = PythonOperator(
        task_id='process_datetime',
        python_callable=process_datetime
    )