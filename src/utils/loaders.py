from src.utils.dirs import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
import pandas as pd
import numpy as np
import datetime


def read_data(folder_type, no_batches=1, read_all=True):
    folders = {
        "raw": RAW_DATA_DIR,
        "interim": INTERIM_DATA_DIR,
        "processed": PROCESSED_DATA_DIR}

    if folder_type not in folders:
        raise Exception(
            f"Invalid folder. Specify one of the following {folders}")

    folder = folders[folder_type]
    folder_files = list(folder.glob("*.csv"))

    if read_all:
        chunked_dataframes = [pd.read_csv(f, index_col=0)
                              for f in folder_files]
        data = pd.concat(chunked_dataframes, ignore_index=True)
    elif no_batches > 1:
        chunked_dataframes = [pd.read_csv(f, index_col=0)
                              for f in folder_files[:no_batches]]
        data = pd.concat(chunked_dataframes, ignore_index=True)
    else:
        data = pd.read_csv(folder_files[0], index_col=0)

    if folder_type in ("raw", "interim"):
        parse_dict_columns(data)

    return data


def read_data_iter(folder_type):
    folders = {
        "raw": RAW_DATA_DIR,
        "interim": INTERIM_DATA_DIR,
        "processed": PROCESSED_DATA_DIR}

    if folder_type not in folders:
        raise Exception(
            f"Invalid folder. Specify one of the following {folders}")

    folder = folders[folder_type]
    folder_files = list(folder.glob("*.csv"))

    for f in folder_files:
        data = pd.read_csv(f, index_col=0)

        if folder_type in ("raw", "interim"):
            parse_dict_columns(data)

        yield data


def parse_dict_columns(data):
    for col in ["parties", "teamA", "teamB"]:
        data[col] = data[col].replace(np.nan, "{}")
        data[col] = data[col].apply(lambda x: eval(x))
