from sklearn.cluster import KMeans
from kneed import KneeLocator


class Cluster:
    def __init__(self, cloud_object, logger=None):
        self.logger = logger
        self.cloud = cloud_object
        self.max_n_clusters = 10
        self.k_clusters = None
        self.save_filename = 'k_means_clustering_model.pkl'
        self.model = None

    def create_cluster(self, features):
        """
        Performs Clustering using K-means++
        Creates KMeans model --> Fits training data --> get the predictions on training set --> Save model on cloud
        :param features:
        :return:
        """
        try:
            k_clusters = self.knee_finder(features)
            self.logger.log_training_pipeline('TRAINING: CLUSTERING : Creating Clusters [algorithm = k-means++]')
            model = KMeans(n_clusters=k_clusters, init='k-means++')
            cluster_label = model.fit_predict(features)
            self.logger.log_training_pipeline('TRAINING: CLUSTERING: Saving Model to Cloud')
            self.cloud.save_model(model, self.save_filename)
            return cluster_label
        except Exception as e:
            self.logger.log_training_pipeline('FAILED: Error Occurred while Creating Clusters')
            raise e

    def knee_finder(self, features):
        """
        Runs KneeLocator from Kneed library to find the optimal number of clusters
        :param features: dataframe containing training data
        :return: Optimal number of clusters
        """
        try:
            self.logger.log_training_pipeline('TRAINING :CLUSTERING: Finding Optimal number of Clusters')
            wcss = []
            for no_of_clusters in range(1, self.max_n_clusters):
                model = KMeans(n_clusters=no_of_clusters, init='k-means++')
                model.fit(features)
                wcss.append(model.inertia_)
            knee_locator = KneeLocator(range(1, self.max_n_clusters), wcss, curve='convex', direction='decreasing')
            self.k_clusters = knee_locator.knee
            self.logger.log_training_pipeline(
                'TRAINING: CLUSTERING : Optimal Number of Clusters Found : [{}]'.format(self.k_clusters)
            )
            return knee_locator.knee
        except Exception as e:
            self.logger.log_training_pipeline('FAILED: Error Occurred while Locating Knee')
            raise e

    def load_clustering_model(self):
        try:
            model = self.cloud.load_model(self.save_filename)
            self.model = model
        except Exception as e:
            print(e)
            # self.logger.pipeline_logs('MODEL LOADING : Error while loading clustering Model')

    def predict_clusters(self, dataframe):
        """
        Predicts cluster number for all records in a dataframe
        :param dataframe:
        :return:
        """
        try:
            self.load_clustering_model()
            if self.model is not None:
                predictions = self.model.predict(dataframe)
                return predictions
            else:
                raise Exception('ClusterPredictionError')
        except Exception as e:
            return False
