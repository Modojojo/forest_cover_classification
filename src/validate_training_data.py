import argparse
import json
from cloud_connect import Cloud
from training_validator import Validator
from fetch_data_from_cloud import read_params, fetch_data
import json


def validate(config_path):
    """
    main function for data validation
    :param config_path:
    :return:
    """
    accepted_data = []
    accepted_filenames = []
    config = read_params(config_path)
    dataframes = fetch_data(config_path)
    for filename in dataframes:
        data = validate_data(filename, dataframes[filename])
        if data is not False:
            accepted_data.append(data)
            accepted_filenames.append(filename)

    validate_data_report_dir = config["reports"]["validate_data_report"]

    with open(validate_data_report_dir, "w") as f:
        report = {
            "accepted_files": accepted_filenames
        }
        json.dump(report, f, indent=4)

    return accepted_data


def validate_data(filename, data):
    """
    Utility function for data validation
    :param filename:
    :param data:
    :return:
    """
    if Validator.validate_file_name(filename):
        if Validator.validate_number_of_columns(data):
            columns = [str(column).lower() for column in data.columns.tolist()]
            data.columns = columns
            if Validator.validate_name_of_columns(data):
                if Validator.validate_null_columns(data):
                    return data
                else:
                    print(f"rejected {filename} due to null column encounter")
                    return False
            else:
                print(f"rejected {filename} due to invalid name of columns")
                return False
        else:
            print(f"rejected {filename} due to invalid Number of columns")
            return False
    else:
        print(f"rejected {filename} due to incorrect filename")
        return False


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    validate(config_path=parsed_args.config)
