from distutils.command.config import config
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime

from src import data_processing, train, utils


def read_config():
    config_path = 'configs/training_config.yaml'
    config = utils.read_config(config_path)
    return config


def train_model(ti):
    config = ti.xcom_pull(task_ids=['read_config'])[0]
    if not config:
        raise Exception('No config file.')
    
    data = utils.load_data(config['train_data'])
    data = data.groupby(by=['shop_id'])
    data = data.get_group(config['shop'])
    dataset = data_processing.TrainDataset(
        data,
        config['features']['features'],
        config['features']['target'],
        config['dataset']['window_size'],
        config['dataset']['start']
    )
    res = train.grid_search(dataset, config)
    return res


def upload_model(ti):
    xcoms = ti.xcom_pull(task_ids=['read_config', 'train_model'])
    config = xcoms[0]
    res = xcoms[1]
    if not config:
        raise Exception('No config file.')
    if not res:
        raise Exception('No res dict.')
    
    train.track_best_model(res, config)
    
    


with DAG(
    dag_id='model_train_dag',
    schedule_interval='@daily',
    start_date=datetime(year=2022, month=10, day=1),
    catchup=False
) as dag:
    # Read config file
    task_read_config = PythonOperator(
        task_id='read_config',
        python_callable=read_config,
        do_xcom_push=True
    )

    # Train model
    task_etl = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        do_xcom_push=True
    )
    
    # Upload best model
    task_upload = PythonOperator(
        task_id='upload_model',
        python_callable=upload_model 
    )
    task_read_config >> task_etl >> task_upload