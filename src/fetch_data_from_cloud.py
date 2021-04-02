import argparse
import yaml
from cloud_connect import Cloud
import json


def read_params(config_path):
    with open(config_path) as config_yaml:
        config = yaml.safe_load(config_yaml)
    return config


def fetch_data(config_path):
    dataframes = {}
    config = read_params(config_path)
    cloud = Cloud(config)
    fetch_report_dir = config["reports"]["fetch_data_report"]

    training_batch_filenames = cloud.get_file_names()

    for filename in training_batch_filenames:
        data = cloud.read_data(filename)
        dataframes[filename] = data

    with open(fetch_report_dir, "w") as f:
        fetch_report = {
            "fetched_files": training_batch_filenames
        }
        json.dump(fetch_report, f, indent=4)

    return dataframes


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    fetch_data(config_path=parsed_args.config)
