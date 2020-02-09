import pandas as pd
from os import path
import configparser
from pymongo import MongoClient


def read_config(section):
    config = configparser.ConfigParser()
    config.read("properties.ini")
    return config[section]


def load_data(load_type, **kwargs):
    if load_type == "json":
        file_path = kwargs.get("filename", None)
        if not path.exists(file_path):
            raise IOError("File {} doesn't exist".format(file_path))
        return pd.read_json(file_path, lines=True)
    elif load_type == "mongoDB":
        mongo_properties = read_config("dev.mongoDB")
        client = MongoClient(
            mongo_properties["ip"],
            int(mongo_properties["port"]),
            username=mongo_properties["username"],
            password=mongo_properties["password"],
        )
        database = client[mongo_properties["database"]]
        collection = database[mongo_properties["collection"]]
        return collection
