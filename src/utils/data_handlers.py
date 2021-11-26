from src.utils.dirs import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR_S, INTERIM_DATA_DIR_S, PROCESSED_DATA_DIR_S
import pandas as pd
import numpy as np
from glob import glob
import datetime
import pickle

DATA_EXT = ('csv', 'feather')

COMPLETE_FOLDERS = {"raw": RAW_DATA_DIR,
                    "interim": INTERIM_DATA_DIR,
                    "processed": PROCESSED_DATA_DIR}
SIMPLIFIED_FOLDERS = {"raw": RAW_DATA_DIR,
                      "interim": INTERIM_DATA_DIR,
                      "processed": PROCESSED_DATA_DIR}


def read_data(folder_type, is_complete, no_batches=None):
    folders = COMPLETE_FOLDERS if is_complete else SIMPLIFIED_FOLDERS

    if folder_type not in folders:
        raise Exception("Invalid folder type. Available: {folders}")

    folder = folders[folder_type]
    folder_files = glob(f'{folder}/*')[:no_batches]
    data_type = folder_files[0].split(".")[-1]

    if data_type == 'csv':
        dfs = [pd.read_csv(f, index_col=0) for f in folder_files]
    elif data_type == 'feather':
        dfs = [pd.read_feather(f) for f in folder_files]
    data = pd.concat(dfs, ignore_index=True)

    if folder_type in ("raw", "interim"):
        parse_dict_columns(data)

    return data


def read_data_iter(folder_type, is_complete):
    folders = COMPLETE_FOLDERS if is_complete else SIMPLIFIED_FOLDERS

    if folder_type not in folders:
        raise Exception("Invalid folder type. Available: {folders}")

    folder = folders[folder_type]
    folder_files = glob(f'{folder}/*')
    data_type = folder_files[0].split(".")[-1]

    for f in folder_files:
        if data_type == 'csv':
            data = pd.read_csv(f, index_col=0)
        elif data_type == 'feather':
            data = pd.read_feather(f)

        if folder_type in ("raw", "interim"):
            parse_dict_columns(data)

        yield data


def parse_dict_columns(data):
    for col in ["parties", "teamA", "teamB"]:
        data[col] = data[col].replace(np.nan, "{}")
        data[col] = data[col].apply(lambda x: eval(x))


def store_processable_match_ids(match_ids, parent_folder):
    match_ids_filename = str(parent_folder) + \
        "\\processable_match_ids.data"

    if match_ids:
        with open(match_ids_filename, 'wb') as f:
            # store the data as binary data stream
            pickle.dump(match_ids, f)


def load_processable_match_ids(parent_folder):
    match_ids_filename = str(parent_folder) + \
        "\\processable_match_ids.data"

    with open(match_ids_filename, 'rb') as f:
        # read the data as binary data stream
        return pickle.load(f)
