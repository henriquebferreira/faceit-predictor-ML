from statistics import mean

import datetime
import numpy as np
from src.features.utils import add_feature


def convert_activated_date_to_timestamp(data):
    def convert_to_utc(utc_dt):
        return utc_dt.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
    date_string = "%a %b %d %X %Z %Y"

    pids_to_timestamp = {}
    for team in ('teamA', 'teamB'):
        for team_data in data[team]:
            for player in team_data:
                if player["id"] not in pids_to_timestamp:
                    activated_at = player["activatedAt"]

                    activated_dt = datetime.datetime.strptime(
                        activated_at, date_string)
                    # convert back to utc
                    activated_utc_dt = convert_to_utc(activated_dt)
                    activated_ts = int(activated_utc_dt.timestamp())
                    pids_to_timestamp[player["id"]] = activated_ts
                else:
                    activated_ts = pids_to_timestamp[player["id"]]

                player["activatedAtTimeStamp"] = activated_ts


def get_mean_faceit_account_age(data, team):
    mean_created_interval = []
    for team_data, start_time in zip(data[team], data["startTime"]):
        accounts_age = [start_time - p["activatedAtTimeStamp"]
                        for p in team_data]
        mean_created_interval.append(sum(accounts_age)/len(accounts_age))

    return mean_created_interval


def get_stddev_faceit_account_age(data, team):
    stddev_created_interval = []
    for team_data, start_time in zip(data[team], data["startTime"]):
        accounts_age = [start_time - p["activatedAtTimeStamp"]
                        for p in team_data]
        stddev_created_interval.append(np.std(accounts_age))
    return stddev_created_interval


def get_min_faceit_account_age(data, team):
    min_created_interval = []
    for team_data, start_time in zip(data[team], data["startTime"]):
        accounts_age = [start_time - p["activatedAtTimeStamp"]
                        for p in team_data]
        min_created_interval.append(min(accounts_age))
    return min_created_interval


def add_date_features(data):
    # TODO: build steam account age features
    convert_activated_date_to_timestamp(data)

    add_feature(data, get_mean_faceit_account_age)
    add_feature(data, get_stddev_faceit_account_age)
    add_feature(data, get_min_faceit_account_age)
