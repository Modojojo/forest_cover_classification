import os
from flask import Flask, render_template, request
from prediction_service.prediction import predict_one_record, prediction, ValueNotEntered
import yaml
from src.training import start_training, read_params
from src.val_insert_training_data_db import validate_and_insert_into_db
import pymongo.errors
from src.custom_logger import Logger
from src.db_connect import DbConnector


params_path = 'params.yaml'
webapp_root = 'webapp'

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        with open("data_schema.yaml") as f:
            data_schema = yaml.safe_load(f)

        wilderness_type = data_schema["wilderness_type"]
        wilderness_type_list = []
        for wilderness in wilderness_type:
            for key, value in wilderness.items():
                wilderness_type_list.append([key, value])

        soil_type = data_schema["soil_type"]
        soil_type_list = []
        for soil in soil_type:
            for key, value in soil.items():
                soil_type_list.append([key, value])

        return render_template("index.html", wilderness_selection=wilderness_type_list, soil_selection=soil_type_list)

    elif request.method == "POST":
        try:
            prediction = predict_one_record(request.form, params_path)
            return render_template("index.html",
                                   prediction="Forest Type Predicted : " + prediction.replace("_", " ").upper())
        except ValueNotEntered as e:
            return render_template("error_page.html", message=str(e))


@app.route("/training_dashboard", methods=["GET"])
def training_dashboard():
    return render_template("training_dashboard.html")


@app.route("/predict", methods=["GET"])
def predict_many():
    try:
        predictions = prediction(params_path)
        return render_template("process_completed.html",
                               message="Batch Data Prediction Process Completed,"
                                       " Predictions have been saved to database")
    except pymongo.errors.ServerSelectionTimeoutError:
        return render_template("error_page.html",
                               message="Could not connect to the Database\nAsk admin to check if Database "
                                       "is running properly")

    except Exception as e:
        return render_template("error_page.html", message=str(e))


@app.route("/startTraining", methods=["POST"])
def perform_training():
    if request.form["accessKey"] == os.environ.get("TRAINING_ACCESS_KEY"):
        try:
            logger = Logger()
            logger.move_logs_to_hist()
            logger.close()
            successful = validate_and_insert_into_db(params_path)
            if successful is True:
                start_training(config_path=params_path)
                return render_template("process_completed.html",
                                       message="Training Process Completed, Please check logs or Metrics")
            else:
                return render_template("error_page.html",
                                       message="Some Error Occurred while - Fetch/Validate/Load Process")
        except pymongo.errors.ServerSelectionTimeoutError:
            return render_template("error_page.html",
                                   message="Cannot connect to the DB, Please check if DB is active or not")

        except Exception as e:
            return render_template("error_page.html",
                                   message=str(e))

    else:
        return render_template("error_page.html",
                               message="Please enter a valid access key to start training, "
                                       "You might not have the access to start Training")


@app.route("/enterAccessKey", methods=["GET"])
def start_training_validator():
    return render_template('start_training.html')


@app.route('/logs', methods=['POST'])
def get_logs():
    try:
        log_collection_name = None
        log_type = request.form.get('logs')
        if log_type == 'training_process':
            log_collection_name = 'training_pipeline'
        elif log_type == 'training':
            log_collection_name = 'training'
        elif log_type == 'file_validation':
            log_collection_name = 'file_validation'

        logger = Logger()
        logs = logger.export_logs(log_collection_name)
        logger.close()
        return render_template('logs.html', logs=logs)
    except pymongo.errors.ServerSelectionTimeoutError:
        return render_template("error_page.html", message=str("Please ask the admin to Check Database Connection"))
    except Exception as e:
        return render_template("error_page.html", message=str(e))


@app.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        config = read_params(params_path)
        db = DbConnector(config)
        metrics = db.fetch_metrics()
        return render_template("metrics.html", metrics=metrics)
    except pymongo.errors.ServerSelectionTimeoutError:
        return render_template("error_page.html", message="Please ask Admin to check the Database Connection")
    except Exception as e:
        return render_template("error_page.html", message=str(e))


if __name__ == "__main__":
    app.run(debug=True)
