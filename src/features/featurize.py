import numpy as np
import pandas as pd

from features.date_features import get_min_created_at_faceit, get_stddev_created_at_faceit, get_mean_created_at_faceit
from features.lifetime_features import get_mean_kd_map_preference, get_mean_winrate_map_preference, \
    get_mean_matches_map_preference, get_mean_kd_on_map, get_mean_winrate_on_map, get_mean_matches_on_map, \
    get_mean_matches, get_mean_winrate, get_mean_kd
from features.match_features import get_mean_elo, get_num_paid_memberships, get_stddev_deviation_elo, \
    get_num_solo_players, get_num_parties, get_num_verified_players
from features.previous_matches_features import get_mean_played_map_on_day, get_mean_matches_on_day, \
    get_mean_first_matches_on_day, get_num_played_togthr_prev, get_mean_delta_elo_prev, get_max_interval_time_prev, \
    get_mean_interval_time_prev, get_mean_delta_rating_prev, get_mean_rating_prev, get_mean_kd_prev, \
    get_mean_assists_prev, get_multikills_score_prev, get_mean_winrate_prev, get_mean_matches_on_map_prev


def add_feature(function, match, **kwargs):
    teams = ("A", "B")

    # returns (everything beyond get_)
    feature = '_'.join(function.__name__.split("_")[1:])

    for team in teams:
        feature_team = '_'.join([feature, team])
        match[feature_team] = function(match, "team" + team, **kwargs)

    match["dif_" + feature] = match[feature + "_A"] - match[feature + "_B"]


def add_date_features(match, players_df):
    players_df['activatedAt_unix'] = pd.DatetimeIndex(players_df['activatedAt']).astype(np.int64) / 1000000000

    add_feature(get_mean_created_at_faceit, match, players=players_df)
    add_feature(get_stddev_created_at_faceit, match, players=players_df)
    add_feature(get_min_created_at_faceit, match, players=players_df)

    # add_feature(get_mean_created_at_steam, match, players=players_df)
    # add_feature(get_stddev_created_at_steam, match, players=players_df)
    # add_feature(get_min_created_at_steam, match, players=players_df)

    return match


def is_player_in_match(match, player_id):
    players_ids = [player['id'] for team in match['teams'] for player in team]
    return player_id in players_ids


def get_players_to_match(match, previous_matches):
    players_to_match = {}
    players_ids = [player['id'] for team in match['teams'] for player in team]
    for player_id in players_ids:
        previous_matches['has_player'] = previous_matches.apply(is_player_in_match, player_id=player_id, axis=1)
        player_previous_matches = previous_matches[previous_matches['has_player'] == True]
        player_previous_matches.sort_values(by='startTime', ascending=False, inplace=True)
        players_to_match[player_id] = player_previous_matches['_id'].head(10).tolist()
    return players_to_match


def add_previous_matches_features(match, previous_matches):
    players_to_match = get_players_to_match(match, previous_matches)

    # previous matches stats might not be accurate if player ELO is different from other team mean ELO
    #  .
    # /_\
    #  |
    # compute corrected stats/rating

    add_feature(get_mean_matches_on_map_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_mean_winrate_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_mean_kd_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_multikills_score_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_mean_rating_prev, match, prevs=previous_matches, p2m=players_to_match)

    add_feature(get_mean_delta_rating_prev, match, prevs=previous_matches, p2m=players_to_match)

    add_feature(get_mean_interval_time_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_mean_interval_time_most_recent_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_max_interval_time_most_recent_prev, match, prevs=previous_matches, p2m=players_to_match)

    add_feature(get_mean_delta_elo_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_mean_dif_rounds_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_mean_dif_elo_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_num_matches_afk, match, prevs=previous_matches, p2m=players_to_match)

    # Players that have played together in the previous matches
    add_feature(get_num_played_togthr_prev, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_winrate_togthr_prev, match, prevs=previous_matches, p2m=players_to_match)

    # Day features
    add_feature(get_mean_first_matches_on_day, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_mean_matches_on_day, match, prevs=previous_matches, p2m=players_to_match)
    add_feature(get_mean_played_map_on_day, match, prevs=previous_matches, p2m=players_to_match)

    return match


def add_lifetime_features(match, players_df):
    add_feature(get_mean_matches, match, players=players_df)
    add_feature(get_mean_winrate, match, players=players_df)
    add_feature(get_mean_kd, match, players=players_df)
    add_feature(get_mean_multikills_score, match, players=players_df)
    #rating includes mvps and assists
    add_feature(get_mean_rating, match, players=players_df)

    add_feature(get_mean_matches_on_map, match, players=players_df)
    add_feature(get_mean_winrate_on_map, match, players=players_df)
    add_feature(get_mean_kd_on_map, match, players=players_df)
    add_feature(get_mean_multikills_on_map, match, players=players_df)
    add_feature(get_mean_rating_on_map, match, players=players_df)

    add_feature(get_mean_matches_map_preference, match, players=players_df)
    add_feature(get_mean_winrate_map_preference, match, players=players_df)
    add_feature(get_mean_kd_map_preference, match, players=players_df)
    add_feature(get_mean_multikills_map_preference, match, players=players_df)
    add_feature(get_mean_rating_map_preference, match, players=players_df)

    add_feature(get_num_verified_players, match, players=players_df)
    add_feature(get_mean_smurf_or_cheater_prob, match, players=players_df)

    return match


def add_match_features(match):
    add_feature(get_mean_elo, match)
    add_feature(get_stddev_deviation_elo, match)
    add_feature(get_num_paid_memberships, match)
    add_feature(get_num_solo_players, match)
    add_feature(get_num_parties, match)

    add_feature(get_entity, match)
    # convert map name to numerical
    # DUMMY MAP

    return match


def add_all_features(match, players, previous_matches, veto_map):
    match['mapPlayed'] = veto_map

    add_match_features(match)
    add_lifetime_features(match, players)
    add_previous_matches_features(match, previous_matches)
    add_date_features(match, players)

    return match


def select_features(match):
    selected_cols = [
        'dif_mean_elo',
        'dif_stddev_deviation_elo',
        'dif_num_paid_memberships',
        'dif_num_solo_players',
        'dif_num_parties',
        'dif_num_new_players',
        'dif_mean_matches',
        'dif_mean_matches_on_map',
        'dif_mean_winrate_on_map',
        'dif_mean_kd_on_map',
        'dif_mean_matches_map_preference',
        'dif_mean_winrate_map_preference',
        'dif_mean_kd_map_preference',
        'dif_mean_matches_on_map_prev',
        'dif_mean_winrate_prev',
        'dif_multikills_score_prev',
        'dif_mean_assists_prev',
        'dif_mean_kd_prev',
        'dif_mean_rating_prev',
        'dif_mean_delta_rating_prev',
        'dif_mean_interval_time_prev',
        'dif_max_interval_time_prev',
        'dif_mean_delta_elo_prev',
        'dif_num_played_togthr_prev',
        'dif_mean_first_matches_on_day',
        'dif_mean_matches_on_day',
        'dif_mean_played_map_on_day',
        'dif_mean_created_at_faceit',
        'dif_stddev_created_at_faceit',
        'dif_min_created_at_faceit'
    ]

    return match[selected_cols]
