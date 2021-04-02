import pymongo
import pandas as pd
import yaml
from datetime import datetime, date

datetime_lib = datetime
date_lib = date


with open("key_utils.yaml") as f:
    key_utils = yaml.safe_load(f)

DB_NAME = key_utils["DB_NAME"]
DB_KEY = key_utils["DB_KEY"]


class DbConnector:
    def __init__(self, config):
        self.TRAINING_COLLECTION_NAME = config["database"]["raw_training_data_db"]
        self.BAD_DATA_COLLECTION_NAME = config["database"]["raw_bad_data_db"]
        self.db_key = DB_KEY
        self.db = None
        self.client = None
        self.database = None
        self.db_name = DB_NAME
        self.datetime_lib = datetime_lib
        self.date_lib = date_lib
        self.connect()

    def connect(self):
        self.client = pymongo.MongoClient(f"mongodb+srv://mododb:{self.db_key}@testcluster.mbnqg.mongodb.net/test")
        self.database = self.client[self.db_name]

    def close(self):
        self.client.close()

    def get_date(self):
        datenow = self.date_lib.today()
        return str(datenow)

    def get_time(self):
        timenow = self.datetime_lib.now()
        current_time = timenow.strftime("%H:%M:%S")
        return str(current_time)

    def insert_training_data(self, file):
        collection = self.database[self.TRAINING_COLLECTION_NAME]
        collection.insert_many(file.to_dict('records'))

    def clear_training_folder(self):
        collection = self.database[self.TRAINING_COLLECTION_NAME]
        collection.drop()

    def fetch_training_data(self):
        collection = self.database[self.TRAINING_COLLECTION_NAME]
        data = pd.DataFrame.from_records(collection.find({}, {'_id':0}))
        return data

    def insert_errored_data(self, file):
        collection = self.database[self.BAD_DATA_COLLECTION_NAME]
        collection.insert_many(file.to_dict('records'))

    def clear_bad_data_folder(self):
        collection = self.database[self.BAD_DATA_COLLECTION_NAME]
        collection.drop()

    def fetch_bad_data(self):
        collection = self.database[self.BAD_DATA_COLLECTION_NAME]
        data = pd.DataFrame.from_records(collection.find({}, {'_id': 0}))
        return data

    # SAVE TRAINING METRICS
    def save_metrics(self, data):
        collection = self.database['METRICS']
        timenow = self.get_time()
        datenow = self.get_date()
        data['date'] = datenow
        data['timestamp'] = timenow
        collection.insert(data)

    def fetch_metrics(self):
        collection = self.database['METRICS']
        return_data = []
        data = collection.find({}, {'_id': 0})
        for doc in data:
            return_data.append(doc)
        return return_data

    # Prediction Functionality

    def insert_predictions(self, predictions):
        collection = self.database['PREDICTIONS']
        for i in predictions:
            predictions[i] = str(predictions[i])
        timenow = self.get_time()
        datenow = self.get_date()
        predictions['date'] = datenow
        predictions['timestamp'] = timenow
        collection.insert(predictions)

    def insert_prediction_data(self, file):
        collection = self.database['prediction_data']
        collection.insert_many(file.to_dict('records'))

    def fetch_prediction_data(self):
        collection = self.database['prediction_data']
        data = pd.DataFrame.from_records(collection.find({}, {'_id':0}))
        return data

    def clear_prediction_folder(self):
        collection = self.database['prediction_data']
        collection.drop()

    def insert_errored_prediction_data(self, file):
        collection = self.database['bad_data_prediction']
        collection.insert_many(file.to_dict('records'))

    def clear_bad_data_prediction_folder(self):
        collection = self.database['bad_data_prediction']
        collection.drop()

    def fetch_bad_data_prediction(self):
        collection = self.database['bad_data_prediction']
        data = pd.DataFrame.from_records(collection.find({}, {'_id': 0}))
        return data
