class Preprocessor:
    def __init__(self, cloud, config, prediction_schema):
        self.cloud = cloud
        self.config = config
        self.prediction_schema = prediction_schema

    def load_standard_scalar(self):
        ss = self.cloud.load_model(self.config["cloud"]["standard_scalar_model"])
        return ss

    def preprocess(self, data):
        return data