class Cluster:
    def __init__(self, cloud_object):
        self.cloud = cloud_object
        self.save_filename = 'k_means_clustering_model.pkl'
        self.model = None

    def load_clustering_model(self):
        model = self.cloud.load_model(self.save_filename)
        self.model = model

    def predict(self, dataframe):
        """
        Predicts cluster number for all records in a dataframe
        :param dataframe:
        :return:
        """
        self.load_clustering_model()
        if self.model is not None:
            predictions = self.model.predict(dataframe)
            return predictions
