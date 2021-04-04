from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np


class Preprocessor:
    def __init__(self, config, logger_object):
        self.logger = logger_object
        self.config = config
        self.target_col = config["base"]["target_col"]
        self.feature_scaling_model = None
        self.dropped_cols = None

    def create_label_encodings(self, data):
        label_encodings = self.config["training_schema"]["target_label_encodings"]
        data[self.target_col] = data[self.target_col].map(label_encodings)
        return data

    def feature_scaling(self, data):
        self.logger.log_training_pipeline("TRAINING: PRE-PROCESSING: Performing Feature Scaling")
        numerical_columns = self.config["training_schema"]["numerical_columns"]
        new_data = data[numerical_columns]
        stdScalar = StandardScaler()
        scaled_data = stdScalar.fit_transform(new_data)
        scaled_data = pd.DataFrame(scaled_data, columns=new_data.columns, index=new_data.index)
        remaining_data = data.drop(numerical_columns, axis=1)
        data = pd.concat([scaled_data, remaining_data], axis=1)
        self.feature_scaling_model = stdScalar
        return data

    def create_features_and_labels(self, data):
        labels = data[self.target_col]
        features = data.drop(self.target_col, axis=1)
        return features, labels

    def oversample_smote(self, features, labels):
        self.logger.log_training_pipeline("TRAINING: PRE-PROCESSING: Performing Oversampling for class re-balance")
        over_sampler = SMOTE()
        features, labels = over_sampler.fit_resample(features, labels)
        return features, labels

    def drop_cols_with_zero_dev(self, dataframe):
        """
        Drops the columns that have a standard Deviation of 0
        :param dataframe:
        :return: Processed Dataframe
        """
        std_devs = np.std(dataframe)
        to_drop = std_devs[std_devs == 0.0].index.tolist()
        if len(to_drop) > 0:
            dataframe = dataframe.drop(to_drop, axis=1)
            print(f"Dropped cols : {to_drop}")
            self.dropped_cols = to_drop
            self.logger.log_training_pipeline(f"TRAINING: PRE-PROCESSING: Dropping non-useful columns: {to_drop}")
        return dataframe

    def preprocess(self, data):
        data = self.create_label_encodings(data)
        data = self.feature_scaling(data)
        data = self.drop_cols_with_zero_dev(data)
        features, labels = self.create_features_and_labels(data)
        features, labels = self.oversample_smote(features, labels)
        return features, labels, self.feature_scaling_model, self.dropped_cols



