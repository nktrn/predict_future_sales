import src.data_processing as dp
import src.utils as ut

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime


def read_config():
    config_path = 'configs/data_config.yaml'
    config = ut.read_config(config_path)
    return config


def clean_data(ti):
    config = ti.xcom_pull(task_ids=['read_config'])[0]
    if not config:
        raise Exception('No config file.')

    sales_df = ut.load_data(config['raw_data']['sales_csv'])
    sales_df = dp.clean_sales_data(sales_df, config['clean_data']['target'], config['clean_data']['percentile'])
    ut.save_data(sales_df, config['cleaned_data']['sales_csv'])


def create_fg(config):
    shop_df = ut.load_data(config['raw_data']['shop_csv'])
    items_df = ut.load_data(config['raw_data']['items_csv'])
    items_cat_df = ut.load_data(config['raw_data']['items_categories_csv'])
    fg = dp.FeatureGenerator(items_df, items_cat_df, shop_df)
    return fg


def process_data(ti):
    xcoms = ti.xcom_pull(task_ids=['read_config'])
    config = xcoms[0]

    if not config:
        raise Exception('No config file.')

    sales_df = ut.load_data(config['cleaned_data']['sales_csv'])

    fg = create_fg(config)

    sales_df = fg.fit_group(sales_df)

    ut.save_data(sales_df, config['processed_data']['sales_csv'])


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

    # Clean data
    task_clean_data = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        do_xcom_push=True
    )

    # Generate features
    task_create_features = PythonOperator(
        task_id='generate_features',
        python_callable=process_data
    )

    task_read_config >> task_clean_data >> task_create_features

