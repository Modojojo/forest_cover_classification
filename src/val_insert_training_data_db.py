import argparse
import json
from training_validator import Validator
from fetch_data_from_cloud import read_params, fetch_data
from db_connect import DbConnector
import json


def validate_and_insert_into_db(config_path):
    """
    main function for data validation and then insertion into database
    :param config_path: params.yaml file path
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

    # prepare the validation report
    validate_data_report_dir = config["reports"]["validate_data_report"]
    with open(validate_data_report_dir, "w") as f:
        report = {
            "accepted_files": accepted_filenames
        }
        json.dump(report, f, indent=4)

    # insert data into database
    n_records_inserted = insert_into_db(accepted_data, config)
    insertion_report_dir = config["reports"]["training_data_insertion_report"]
    with open(insertion_report_dir, "w") as f:
        report = {
            "number_of_documents_inserted": n_records_inserted
        }
        json.dump(report, f, indent=4)

    return True


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
                    try:
                        features = data.drop('class', axis=1)
                        labels = data['class']
                        features = features.astype(int)
                        features.insert(len(features.columns), 'class', labels)
                        data = features
                        return data
                    except Exception as e:
                        print(f"rejected {filename} : Not able to convert column data type to int")
                        return False
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


def insert_into_db(dataframes, config):
    n_records_inserted = 0
    db = DbConnector(config)
    db.clear_training_folder()
    for data in dataframes:
        db.insert_training_data(data)
        n_records_inserted = n_records_inserted + len(data)
    db.close()
    return n_records_inserted


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    validate_and_insert_into_db(config_path=parsed_args.config)
