import pandas as pd


class Preprocessor:
    def __init__(self, cloud, config, prediction_schema):
        self.cloud = cloud
        self.config = config
        self.prediction_schema = prediction_schema

    def load_standard_scalar(self):
        ss = self.cloud.load_model(self.config["cloud"]["standard_scaler_model"])
        return ss

    def drop_unwanted_cols(self, data):
        unwanted_cols = self.prediction_schema["dropped_columns"]
        cols_to_drop = []
        for col in unwanted_cols:
            if col != "":
                cols_to_drop.append(col)
        data = data.drop(cols_to_drop, axis=1)
        return data

    def perform_scaling(self, data):
        ss = self.load_standard_scalar()
        cols_to_scale = self.config["prediction_schema"]["numerical_columns"]
        data_to_scale = data[cols_to_scale]
        remaining_data = data.drop(cols_to_scale, axis=1)

        scaled_data = ss.transform(data_to_scale)

        scaled_data = pd.DataFrame(scaled_data, columns=data_to_scale.columns, index=data_to_scale.index)

        data = pd.concat([scaled_data, remaining_data], axis=1)
        return data

    def preprocess(self, data):
        data = self.drop_unwanted_cols(data)
        data = self.perform_scaling(data)
        return data
