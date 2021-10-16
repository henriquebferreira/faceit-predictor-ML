from statistics import mean

import numpy as np

from features.utils import get_team_rounds


def get_mean_elo(match, team):
    elos = [player['elo'] for player in match[team]]
    return mean(elos)


def get_stddev_deviation_elo(match, team):
    elos = [player['elo'] for player in match[team]]
    return np.std(elos)


def get_num_paid_memberships(match, team):
    memberships = [player['membership'] for player in match[team] if player['membership'] != 'free']
    return len(memberships)


def get_num_solo_players(match, team):
    num_solo_players = 0
    if not match['parties']:
        return 0

    team_players_ids = [p['id'] for p in match[team]]
    for _, party_players_ids in match["parties"].items():
        if party_players_ids[0] in team_players_ids:
            # Increment number of solo players if it belongs to a party with size 1
            if len(party_players_ids) == 1:
                num_solo_players += 1
    return num_solo_players


def get_num_parties(match, team):
    num_parties = 0
    if not match['parties']:
        return 1

    team_players_ids = [p['id'] for p in match[team]]
    for _, party_players_ids in match["parties"].items():
        if party_players_ids[0] in team_players_ids:
            num_parties += 1
    return num_parties


def get_num_verified_players(match, team, **kwargs):
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    verified_values = players_df[players_df['_id'].isin(team_players_ids)]['verified'].values
    num_verified_players = sum(verified_values)
    return num_verified_players


def get_winner(match):
    team_rounds = get_team_rounds(match['score'])
    return 1 if team_rounds[0] > team_rounds[1] else 0
