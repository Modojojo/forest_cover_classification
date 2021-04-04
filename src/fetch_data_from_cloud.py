import argparse
import yaml
import json
try:
    from cloud_connect import Cloud
    from custom_logger import Logger
except Exception:
    from src.cloud_connect import Cloud
    from src.custom_logger import Logger

def read_params(config_path):
    with open(config_path) as config_yaml:
        config = yaml.safe_load(config_yaml)
    return config


def fetch_data(config_path):
    logger = Logger()
    dataframes = {}
    config = read_params(config_path)
    cloud = Cloud(config)
    fetch_report_dir = config["reports"]["fetch_data_report"]

    logger.log_training_pipeline("FETCH DATA: Fetching Names of training files from server")
    training_batch_filenames = cloud.get_file_names()

    logger.log_training_pipeline("FETCH DATA: Reading available files from server")
    for filename in training_batch_filenames:
        data = cloud.read_data(filename)
        dataframes[filename] = data

    with open(fetch_report_dir, "w") as f:
        fetch_report = {
            "fetched_files": training_batch_filenames
        }
        json.dump(fetch_report, f, indent=4)

    logger.close()
    return dataframes


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    fetch_data(config_path=parsed_args.config)
