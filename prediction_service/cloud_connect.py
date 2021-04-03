import pandas as pd
from boto3.session import Session
import pickle
import json
import os


access_key = os.environ.get("CLOUD_ACCESS_KEY")
secret_access_key = os.environ.get("CLOUD_SECRET_ACCESS_KEY")


class Cloud:
    def __init__(self, config):
        self.access_key = access_key
        self.secret_access_key = secret_access_key
        self.session = None
        self.s3_resource = None
        self.bucket_name = config["cloud"]["bucket_name"]
        self.bucket = None
        self.connect()
        self.training_data_dir = config["cloud"]["training_data_folder"]
        self.prediction_data_dir = config["cloud"]["prediction_data_folder"]
        self.models_path = config["cloud"]["models_save_path"]

    def connect(self):
        """
        Connects to the s3 bucket
        :return:
        """
        try:
            self.session = Session(aws_access_key_id=access_key, aws_secret_access_key=secret_access_key)
            self.s3_resource = self.session.resource('s3')
            self.bucket = self.s3_resource.Bucket(self.bucket_name)
        except Exception as e:
            raise Exception('Some Error occurred while connecting to the cloud storage')
        return

    def read_data(self, filename, predicton=False):
        """
        Reads the data file using pandas
        :param path: complete path to the s3 file
        :return:
        """
        try:
            if predicton is False:
                resource = self.s3_resource
                s3_object = resource.Object(self.bucket_name, str(self.training_data_dir) + str(filename))
                object_response = s3_object.get()
                data = pd.read_csv(object_response['Body'])
                return data
            else:
                resource = self.s3_resource
                s3_object = resource.Object(self.bucket_name, str(self.prediction_data_dir) + str(filename))
                object_response = s3_object.get()
                data = pd.read_csv(object_response['Body'])
                return data
        except Exception as e:
            raise Exception("Error while reading the file : {}".format(filename))

    def get_file_names(self, prediction=False):
        """
        Returns a list of names of all the files in a S3 Folder
        :return: List of all the Filenames
        """
        if prediction is False:
            filename_list = []
            for objects in self.bucket.objects.filter(Prefix=self.training_data_dir):
                filename = str(objects.key).split('/')[-1]
                if filename != "":
                    filename_list.append(filename)
            return filename_list
        else:
            filename_list = []
            for objects in self.bucket.objects.filter(Prefix=self.prediction_data_dir):
                filename = str(objects.key).split('/')[-1]
                if filename != "":
                    filename_list.append(filename)
            return filename_list

    def save_model(self, model, filename):
        pickle_object = pickle.dumps(model)
        self.s3_resource.Object(self.bucket_name, self.models_path + str(filename)).put(Body=pickle_object)
        return

    def load_model(self, filename):
        """
        Loads Models from cloud saved in directory - wafer/models/<filename>
        :param filename:
        :return:
        """
        model_object = self.s3_resource.Object(self.bucket_name, self.models_path + str(filename)).get()['Body'].read()
        model = pickle.loads(model_object)
        return model

    def write_json(self, json_file, filename):
        json_object = json.dumps(json_file)
        self.s3_resource.Object(self.bucket_name, self.models_path + str(filename)).put(Body=json_object)
        return

    def load_json(self, filename):
        json_object = self.s3_resource.Object(self.bucket_name, self.models_path + str(filename)).get()['Body'].read()
        json_file = json.loads(json_object)
        return json_file

