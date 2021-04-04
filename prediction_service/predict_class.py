class Classifier:
    def __init__(self, config, cloud, prediction_schema):
        self.config = config
        self.cloud = cloud
        self.schema = prediction_schema

    def load_model(self, cluster_number):
        model_name = self.schema[str(cluster_number)]
        model = self.cloud.load_model(model_name)
        return model

    def load_required_models(self, cluster_numbers):
        models = {}
        for cluster in cluster_numbers:
            model = self.load_model(cluster)
            models[str(cluster)] = model
        return models

    def predict(self, data, clusters):
        final_predictions = {}
        unique_cluster_numbers = list(set(clusters))
        models = self.load_required_models(unique_cluster_numbers)
        index_ = list(range(len(data)))
        data.insert(0, "index", index_)
        data['cluster'] = clusters
        for cluster in unique_cluster_numbers:
            features = data[data["cluster"] == cluster].drop(["cluster"], axis=1)
            indexes = features["index"].tolist()
            features = features.drop(["index"], axis=1)

            model = models[str(cluster)]

            predictions = model.predict(features)

            for i in range(len(predictions)):
                final_predictions[indexes[i]] = predictions[i]

            temp_predictions = final_predictions
            final_predictions = {}
            for i in sorted(temp_predictions):
                final_predictions[i] = temp_predictions[i]

        return final_predictions


