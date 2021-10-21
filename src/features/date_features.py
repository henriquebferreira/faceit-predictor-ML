from statistics import mean

import datetime
import numpy as np
from src.features.utils import add_feature


def convert_activated_date_to_timestamp(data):
    def convert_to_utc(utc_dt):
        return utc_dt.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
    date_string = "%a %b %d %X %Z %Y"

    for team in ('teamA', 'teamB'):
        for team_data in data[team]:
            for player in team_data:
                activated_at = player["activatedAt"]

                activated_datetime = datetime.datetime.strptime(
                    activated_at, date_string)
                # convert back to utc
                activated_utc_datetime = convert_to_utc(activated_datetime)
                activated_timestamp = int(activated_utc_datetime.timestamp())

                player["activatedAtTimeStamp"] = activated_timestamp


def get_mean_created_at_faceit(data, team):
    mean_created_interval = []
    for team_data, start_time in zip(data[team], data["startTime"]):
        team_mean_account_age = mean(
            [start_time - p["activatedAtTimeStamp"] for p in team_data])
        mean_created_interval.append(team_mean_account_age)
    return mean_created_interval


def get_stddev_created_at_faceit(data, team):
    stddev_created_interval = []
    for team_data, start_time in zip(data[team], data["startTime"]):
        team_stddev_account_age = np.std(
            [start_time - p["activatedAtTimeStamp"] for p in team_data])
        stddev_created_interval.append(team_stddev_account_age)
    return stddev_created_interval


def get_min_created_at_faceit(data, team):
    min_created_interval = []
    for team_data, start_time in zip(data[team], data["startTime"]):
        team_min_account_age = min(
            [start_time - p["activatedAtTimeStamp"] for p in team_data])
        min_created_interval.append(team_min_account_age)
    return min_created_interval


def add_date_features(data):
    convert_activated_date_to_timestamp(data)

    add_feature(data, get_mean_created_at_faceit)
    add_feature(data, get_stddev_created_at_faceit)
    add_feature(data, get_min_created_at_faceit)
