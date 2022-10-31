import yaml
import pandas as pd


def clean_data(df, target, percentile):
    df = df.drop_duplicates()
    perc = df[target].quantile(percentile)
    df = df[df[target] <= perc]
    return df


def load_data(path):
    return pd.read_csv(path)


def save_data(df, path):
    df.to_csv(path, index=False)


def read_config(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
