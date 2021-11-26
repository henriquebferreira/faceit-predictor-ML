from pymongo import MongoClient
from src.db.config import read_config


def get_local_db():
    db_cfg = read_config("local.ingestorDB")
    client = MongoClient(**db_cfg)
    db = client['faceit_imported']

    return db
