from sklearn.model_selection import train_test_split
import argparse
import yaml
import json
try:
    from db_connect import DbConnector
    from model_builder import Model
    from cluster_builder import Cluster
    from cloud_connect import Cloud
    from training_data_preprocessor import Preprocessor
    from custom_logger import Logger
except Exception:
    from src.db_connect import DbConnector
    from src.model_builder import Model
    from src.cluster_builder import Cluster
    from src.cloud_connect import Cloud
    from src.training_data_preprocessor import Preprocessor
    from src.custom_logger import Logger


def read_params(config_path):
    with open(config_path) as config_yaml:
        config = yaml.safe_load(config_yaml)
    return config


def fetch_training_data(db_object):
    try:
        training_data = db_object.fetch_training_data()
        return training_data
    except Exception as e:
        raise Exception("FAILED : fetch_training_data : not able to fetch training data from db")


def start_training(config_path):
    config = read_params(config_path)

    logger = Logger()
    cloud = Cloud(config)
    db = DbConnector(config)

    logger.log_training_pipeline("TRAINING: STARTED")

    # FETCHING DATA FROM DB
    logger.log_training_pipeline("TRAINING: Fetching training data from database")
    training_data = fetch_training_data(db)

    # PREPROCESS DATA
    logger.log_training_pipeline("TRAINING: PREPROCESSING: Processing training data fetched from db ")
    preprocessor = Preprocessor(config, logger)
    features, labels, standardScalerModel, dropped_cols = preprocessor.preprocess(training_data)

    # SAVING STANDARD SCALER MODEL TO CLOUD
    logger.log_training_pipeline("TRAINING: Saving feature scaling model to cloud")
    cloud.save_model(standardScalerModel, config["cloud"]["standard_scaler_model"])

    # CLUSTERING
    logger.log_training_pipeline("TRAINING: CLUSTERING: STARTED")
    cluster_builder = Cluster(cloud, logger=logger)
    cluster_id = cluster_builder.create_cluster(features=features)
    # Add cluster column in features - to be used while training individual models for individual clusters
    features.insert(loc=len(features.columns),
                    column="cluster",
                    value=cluster_id)

    # COMBINING LABELS AND FEATURES AS ONE DATAFRAME
    training_data = features
    training_data.insert(loc=len(training_data.columns),
                         column=config["base"]["target_col"],
                         value=labels)

    prediction_schema_dict = {"dropped_columns": dropped_cols}  # will be used to store model and cluster relations

    logger.log_training_pipeline("TRAINING CLASSIFIER: STARTED, please check Training Logs for detailed information")
    # MODEL TRAINING
    for cluster_number in training_data["cluster"].unique().tolist():

        logger.log_training("PROCESS STARTED")
        logger.log_training(f"Started training for cluster {[cluster_number]}")

        # fetch data based on cluster number and divide into training and testing sets
        data = training_data[training_data["cluster"] == cluster_number].drop(["cluster"], axis=1)
        training_features = data.drop(config["base"]["target_col"], axis=1)
        training_labels = data[config["base"]["target_col"]]
        x_train, x_test, y_train, y_test = train_test_split(training_features,
                                                            training_labels,
                                                            test_size=config["training_schema"]["test_size"])

        # CREATE MODEL_BUILDER OBJECT, TRAIN MODELS AND OBTAIN THE BEST MODEL
        model = Model(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test, logger_object=logger)
        (best_model, best_model_name, best_model_metrics) = model.get_best_model()

        # Create model filepath for cloud storage
        logger.log_training(f"Saving {best_model_name} model to cloud")
        model_filename = str(cluster_number) + '_' + str(best_model_name) + '/' + 'model.pkl'
        cloud.save_model(best_model, model_filename)    # Save model to cloud

        logger.log_training(f"Saving performance metrics for the recent model trained")
        db.save_metrics(best_model_metrics)     # Save trained model metrics in database
        prediction_schema_dict[str(cluster_number)] = model_filename    # saving the model related to current cluster no

    logger.log_training("Saving PREDICTION_SCHEMA to cloud")
    cloud.write_json(prediction_schema_dict, "prediction_schema.json")  # writing prediction schema file to cloud
    training_models_report = config["reports"]["training_models_report"]    # fetch reports directory path
    with open(training_models_report, "w") as f:
        json.dump(prediction_schema_dict, f, indent=4)      # save the prediction_schema info in reports

    logger.log_training_pipeline("TRAINING CLASSIFIER: COMPLETED")

    # close connections
    db.close()
    logger.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    start_training(config_path=parsed_args.config)
