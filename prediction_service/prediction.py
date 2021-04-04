import argparse
import yaml
import pandas as pd
import numpy as np
import re

# Adding try catch block to import modules
# reason - because dvc repro runs -- python prediction_service/predictions.py
#          in that we cannot import modules from prediction service, instead directly importing is Done

try:
    from prediction_service.db_connect import DbConnector
    from prediction_service.prediction_preprocessor import Preprocessor
    from prediction_service.cloud_connect import Cloud
    from prediction_service.predict_cluster import Cluster
    from prediction_service.predict_class import Classifier
except Exception as e:
    from db_connect import DbConnector
    from prediction_preprocessor import Preprocessor
    from cloud_connect import Cloud
    from predict_cluster import Cluster
    from predict_class import Classifier


def read_params(config_path):
    with open(config_path) as config_yaml:
        config = yaml.safe_load(config_yaml)
    return config


def prediction(config_path, data=False):
    print("started predictions")
    config = read_params(config_path)
    db = DbConnector(config)
    cloud = Cloud(config)
    prediction_schema = cloud.load_json(config["cloud"]["prediction_schema"])

    # FETCH DATA FROM DB
    if data is False:
        non_processed_data = db.fetch_prediction_data()
        print("loaded data")
    else:
        non_processed_data = data

    # PREPROCESS DATA
    preprocessor = Preprocessor(cloud, config, prediction_schema)
    processed_data = preprocessor.preprocess(non_processed_data)
    print("preprocessing completed")

    # CLUSTERING
    cluster_object = Cluster(cloud_object=cloud)
    cluster_predictions = cluster_object.predict(processed_data)
    print("clustering_completed")

    # CLASSIFICATION
    classifier = Classifier(config=config,
                            cloud=cloud,
                            prediction_schema=prediction_schema)
    predictions = classifier.predict(data=processed_data, clusters=cluster_predictions)
    print("classification completed")

    # SAVE PREDICTIONS IN DB
    if data is False:
        db.insert_predictions(predictions)
        print("saving predictions to DB")
        db.close()
        return True
    else:
        db.close()
        return predictions


def predict_one_record(form, config_path):
    values = [form["elevation"],
              form["aspect"],
              form["slope"],
              form["horizontal_distance_to_hydrology"],
              form["vertical_distance_to_hydrology"],
              form["horizontal_distance_to_roadways"],
              form["horizontal_distance_to_fire_points"]]
    wilderness_type = form["wildernessType"]
    soil_type = form["soilType"]
    soil_type_column_name = "soil_type_" + str(soil_type)
    wilderness_type_column_name = "wilderness_area" + str(wilderness_type)

    # prepare for prediction:

    config = read_params(config_path)
    columns = config["prediction_schema"]["column_names"]
    data = np.zeros(len(columns)).astype(int)
    df = pd.DataFrame([data], columns=columns)
    df[config["prediction_schema"]["numerical_columns"]] = values
    df[[wilderness_type_column_name]] = 1
    df[[soil_type_column_name]] = 1

    # check if any empty column is there:
    for value in values:
        if value == "":
            raise ValueNotEntered
    if not re.match(r"^soil_type_[0-9]{1,2}$", soil_type_column_name):
        raise ValueNotEntered
    if not re.match(r"^wilderness_area[0-9]{1}$", wilderness_type_column_name):
        raise ValueNotEntered

    # Predict
    prediction_ = prediction(config_path=config_path, data=df)
    predicted_class = prediction_[0]
    class_label = config["prediction_schema"]["target_label_encodings"][predicted_class]
    return class_label


class ValueNotEntered(Exception):
    def __init__(self, message="Values Not Entered Properly"):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    prediction(config_path=parsed_args.config)
