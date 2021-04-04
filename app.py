import os
from flask import Flask, render_template, request
from prediction_service.prediction import predict_one_record, prediction
import yaml

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
        prediction = predict_one_record(request.form, params_path)
        return render_template("index.html",
                               prediction="Forest Type Predicted : " + prediction.replace("_", " ").upper())


@app.route("/training_dashboard", methods=["GET"])
def training_dashboard():
    return render_template("training_dashboard.html")


@app.route("/predict", methods=["GET"])
def predict_many():
    print("prediction batch started")
    predictions = prediction(params_path)
    return render_template("process_completed.html",
                           message="Batch Data Prediction Process Completed, Predictions have been saved to database")


if __name__ == "__main__":
    app.run(debug=True)
