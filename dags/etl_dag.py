import src.data_processing as dp

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime


def read_config():
    config_path = 'configs/data_config.yaml'
    config = dp.read_config(config_path)
    return config


def process_data(ti):
    config = ti.xcom_pull(task_ids=['read_config'])[0]
    if not config:
        raise Exception('No config file.')

    sales_df = dp.load_data(config['raw_data']['sales_csv'])
    items_df = dp.load_data(config['raw_data']['items_csv'])
    items_cat_df = dp.load_data(config['raw_data']['items_categories_csv'])
    shop_df = dp.load_data(config['raw_data']['shop_csv'])

    sales_df = dp.clean_data(sales_df, config['clean_data']['target'], config['clean_data']['percentile'])

    fg = dp.FeatureGenerator(items_df, items_cat_df, shop_df)
    sales_df = fg.fit_group(sales_df)

    if config['group_data']['group']:
        sales_df = dp.group_data(sales_df, config['group_data']['by'], config['group_data']['group_number'])
    
    dp.save_data(sales_df, config['processed_data']['sales_csv'])
    return 1


with DAG(
    dag_id='etl_dag',
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

    # ETL
    task_etl = PythonOperator(
        task_id='do_etl',
        python_callable=process_data
    )
    task_read_config >> task_etl

