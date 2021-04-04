import pymongo
import pymongo.errors
from datetime import datetime, date
import os

datetime_lib = datetime
date_lib = date

DB_NAME = os.environ.get("DB_NAME")
DB_KEY = os.environ.get("DB_KEY")


class Logger:
    def __init__(self):
        self.db = None
        self.client = None
        self.database = None
        self.DB_NAME = DB_NAME
        self.DB_KEY = DB_KEY
        self.datetime_lib = datetime_lib
        self.date_lib = date_lib
        self.connect()

    def connect(self):
        """
        Connect to the mongoDb database
        :return:
        """
        try:
            self.client = pymongo.MongoClient(f"mongodb+srv://mododb:{self.DB_KEY}@testcluster.mbnqg.mongodb.net/test")
            self.database = self.client[self.DB_NAME]
            self.client.server_info()
            return True
        except pymongo.errors.ServerSelectionTimeoutError as e:
            raise e

    def close(self):
        self.client.close()

    def get_date(self):
        datenow = self.date_lib.today()
        return str(datenow)

    def get_time(self):
        timenow = self.datetime_lib.now()
        current_time = timenow.strftime("%H:%M:%S")
        return str(current_time)

    def move_logs_to_hist(self):
        collections_list = ['raw_validation_logs', 'training_pipeline_logs', 'training_logs']
        move_to_collection_list = ['hist_raw_validation_logs', 'hist_pipeline_logs', 'hist_training_logs']
        for i in range(len(collections_list)):
            from_collection = self.database[collections_list[i]]
            to_collection = self.database[move_to_collection_list[i]]
            for row in from_collection.find({}, {'_id': 0}):
                to_collection.insert(row)
            from_collection.drop()

    def export_logs(self, log_type):
        """
        Export the logs for a particular collection (log_type)
        :param log_type: ['file_validation', 'data_validation', 'training', 'prediction']
        :return:
        """
        logfile = []
        collection = None

        if log_type == 'file_validation':
            collection = self.database['raw_validation_logs']

        elif log_type == 'training':
            collection = self.database['training_logs']

        elif log_type == 'prediction':
            collection = self.database['prediction_logs']

        elif log_type == 'training_pipeline':
            collection = self.database['process_logs']

        if collection is not None:
            for row in collection.find({}, {'_id': 0}):
                logfile.append(row)

        return self.format_logs(logfile)

    def format_logs(self, logfile):
        """
        Utility function for formatting the log file while exporting
        :param logfile: List of all the logs in dictionary format
        :return:
        """
        log_file_export = []
        for row in logfile:
            log_list = [row[i] for i in row]
            log_file_export.append(log_list)
        return log_file_export

    def log_file_validation(self,  message):
        """
        Logs File validation logs
        :param message: Error Message
        :return:
        """
        collection = self.database['raw_validation_logs']
        timenow = self.get_time()
        datenow = self.get_date()
        i_datetime = str(datenow) + ' ' + str(timenow)
        insert_dict = {"datetime": i_datetime,
                       'error': message}
        try:
            collection.insert(insert_dict)
        except pymongo.errors.ServerSelectionTimeoutError as e:
            raise e
        except Exception:
            raise Exception('Logger Failed, Please check database connection')

    def log_training_pipeline(self,  message):
        """
        Complete Process Logging
        :param message: Message
        :return:
        """
        collection = self.database['process_logs']
        timenow = self.get_time()
        datenow = self.get_date()
        i_datetime = str(datenow) + ' ' + str(timenow)
        insert_dict = {"datetime": i_datetime,
                       'message': message}
        try:
            collection.insert(insert_dict)
        except pymongo.errors.ServerSelectionTimeoutError as e:
            raise e
        except Exception:
            raise Exception('Logger Failed, Please check database connection')

    def log_training(self, message):
        """
        Complete Process Logging
        :param message: Message
        :return:
        """
        collection = self.database['training_logs']
        timenow = self.get_time()
        datenow = self.get_date()
        i_datetime = str(datenow) + ' ' + str(timenow)
        insert_dict = {"datetime": i_datetime,
                       'message': message}
        try:
            collection.insert(insert_dict)
        except pymongo.errors.ServerSelectionTimeoutError as e:
            raise e
        except Exception:
            raise Exception('Logger Failed, Please check database connection')
