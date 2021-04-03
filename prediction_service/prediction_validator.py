import re
import yaml

with open("params.yaml") as f:
    config = yaml.safe_load(f)

training_schema = config["prediction_schema"]
NUMBER_OF_COLUMNS = training_schema["number_of_columns"]
COLUMN_NAMES = training_schema["column_names"]


class Validator:

    @staticmethod
    def validate_file_name(filename):
        """
        Perform Filename and filetype Validation
        :param filename: Exact Name of the file with extension
        :return: True if Filename and type are valid, else raise Exception
        """
        pattern = r'^forest_cover_[0-9]{8}_[0-9]{6}\.csv$'
        if not re.match(pattern, str(filename).lower()):
            return False
        else:
            return True

    @staticmethod
    def validate_number_of_columns(df):
        """
        Validate the number of columns in the dataset
        :param df: Pandas Dataframe
        :return: True if valid, else Raise Exception
        """
        if not len(df.columns) == NUMBER_OF_COLUMNS:
            return False
        else:
            return True

    @staticmethod
    def validate_name_of_columns(df):
        columns = df.columns
        if columns.tolist() == COLUMN_NAMES:
            return True
        else:
            return False
