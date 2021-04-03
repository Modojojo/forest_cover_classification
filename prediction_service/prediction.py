import argparse
import yaml
from db_connect import DbConnector
from prediction_preprocessor import Preprocessor
from cloud_connect import Cloud
from predict_cluster import Cluster


def read_params(config_path):
    with open(config_path) as config_yaml:
        config = yaml.safe_load(config_yaml)
    return config


def prediction(config_path):
    config = read_params(config_path)
    db = DbConnector(config)
    cloud = Cloud(config)
    prediction_schema = cloud.load_json(config["cloud"]["prediction_schema"])

    # FETCH DATA FROM DB
    non_processed_data = db.fetch_prediction_data()

    # PREPROCESS DATA
    preprocessor = Preprocessor(cloud, config, prediction_schema)
    processed_data = preprocessor.preprocess(non_processed_data)

    # CLUSTERING
    cluster_object = Cluster(cloud_object=cloud)
    cluster_predictions = cluster_object.predict(processed_data)

    # # CLASSIFICATION
    # predictions = predict_class(processed_data, cluster_predictions)
    #
    # # SAVE PREDICTIONS IN DB
    # db.insert_predictions(predictions)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    prediction(config_path=parsed_args.config)
