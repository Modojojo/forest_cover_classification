from cloud_connect import Cloud
from db_connect import DbConnector
from prediction_validator import Validator
import yaml
import argparse
import json


def read_params(config_path):
    with open(config_path) as config_yaml:
        config = yaml.safe_load(config_yaml)
    return config


def fetch_validate_load(config_path):
    config = read_params(config_path)
    db = DbConnector(config)
    data = fetch_data(config)
    accepted_data = validate_data(data)
    db.clear_prediction_folder()
    inserted = 0
    for filename in accepted_data:
        inserted = inserted + len(accepted_data[filename])
        db.insert_prediction_data(accepted_data[filename])
        print(f"inserted {filename}")

    insertion_report_dir = config["reports"]["prediction_data_insertion_reports"]
    with open(insertion_report_dir, "w") as f:
        report = {
            "number_of_documents_inserted": inserted
        }
        json.dump(report, f, indent=4)
    db.close()


def fetch_data(config):
    cloud = Cloud(config)
    filenames = cloud.get_file_names(prediction=True)
    data = {}
    for file in filenames:
        data[file] = cloud.read_data(file, predicton=True)
    return data


def validate_data(main_data):
    accepted_data = {}
    for filename in main_data:
        data = main_data[filename]
        if Validator.validate_file_name(filename):
            if Validator.validate_number_of_columns(data):
                columns = [str(column).lower() for column in data.columns.tolist()]
                data.columns = columns
                if Validator.validate_name_of_columns(data):
                    try:
                        data = data.astype(int)
                        accepted_data[filename] = data
                    except Exception as e:
                        print(f"rejected {filename} : Not able to convert column data type to int")
                else:
                    print(f"rejected {filename} due to invalid name of columns")
            else:
                print(f"rejected {filename} due to invalid Number of columns")
        else:
            print(f"rejected {filename} due to incorrect filename")
    return accepted_data


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    fetch_validate_load(config_path=parsed_args.config)





