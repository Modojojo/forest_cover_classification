import os
from flask import Flask, render_template, request
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
        elevation = request.form["elevation"]
        aspect = request.form["aspect"]
        slope = request.form["slope"]
        horizontal_distance_to_hydrology = request.form["horizontal_distance_to_hydrology"]
        vertical_distance_to_hydrology = request.form["vertical_distance_to_hydrology"]
        horizontal_distance_to_roadways = request.form["horizontal_distance_to_roadways"]
        horizontal_distance_to_fire_points = request.form["horizontal_distance_to_fire_points"]
        return render_template("index.html", prediction=request.form)


@app.route("/training", methods=["GET"])
def training_dashboard():
    return render_template("training_dashboard.html")


if __name__ == "__main__":
    app.run(debug=True)
