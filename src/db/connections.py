from pymongo import MongoClient
from src.db.config import read_config


def get_local_ingestor_db():
    db_cfg = read_config("local.ingestorDB")
    client = MongoClient(**db_cfg)
    db = client['faceit_imported']

    return db


def get_staging_db():
    db_cfg = read_config("staging.atlasDB")
    username = db_cfg["username"]
    password = db_cfg["password"]
    srv_connection = db_cfg["srv_connection"]
    connection_string = f"mongodb+srv://{username}:{password}@{srv_connection}"
    client = MongoClient(connection_string)

    db = client['faceit-predictor']
    return db


def get_prod_db():
    db_cfg = read_config("prod.atlasDB")
    username = db_cfg["username"]
    password = db_cfg["password"]
    srv_connection = db_cfg["srv_connection"]
    connection_string = f"mongodb+srv://{username}:{password}@{srv_connection}"
    client = MongoClient(connection_string)

    db = client['faceit-predictor']
    return db
