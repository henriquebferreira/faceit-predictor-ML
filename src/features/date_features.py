from statistics import mean

import numpy as np


def get_mean_created_at_faceit(match, team, **kwargs):
    date_match = match['startTime']
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    activation_times = players_df[players_df['_id'].isin(team_players_ids)]['activatedAt_unix'].values
    date_intervals = [date_match - activation_time for activation_time in activation_times]
    return mean(date_intervals)


def get_stddev_created_at_faceit(match, team, **kwargs):
    date_match = match['startTime']
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    activation_times = players_df[players_df['_id'].isin(team_players_ids)]['activatedAt_unix'].values
    date_intervals = [date_match - activation_time for activation_time in activation_times]
    return np.std(date_intervals)


def get_min_created_at_faceit(match, team, **kwargs):
    date_match = match['startTime']
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    activation_times = players_df[players_df['_id'].isin(team_players_ids)]['activatedAt_unix'].values
    date_intervals = [date_match - activation_time for activation_time in activation_times]
    return min(date_intervals)
