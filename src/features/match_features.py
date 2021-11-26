from src.features.utils import get_team_rounds
from statistics import mean

import numpy as np
import pandas as pd
from src.features.utils import add_feature


def get_mean_elo(data, team):
    mean_elos = []
    for team_data in data[team]:
        team_mean_elo = mean([player["elo"] for player in team_data])
        mean_elos.append(team_mean_elo)
    return mean_elos


def get_stddev_elo(data, team):
    stddev_elos = []
    for team_data in data[team]:
        team_stddev_elo = np.std([player["elo"] for player in team_data])
        stddev_elos.append(team_stddev_elo)
    return stddev_elos


def get_num_paid_memberships(data, team):
    memberships = []
    for team_data in data[team]:
        membership = len([p['membership']
                          for p in team_data if p['membership'] != 'free'])
        memberships.append(membership)
    return memberships


def get_num_solo_players(data, team):
    solo_players = []

    for parties_data, team_data in zip(data["parties"], data[team]):
        if not parties_data:
            solo_players.append(0)
            continue

        num_solo_players = 0
        team_players_ids = [p['id'] for p in team_data]

        for party_players_ids in parties_data.values():
            if len(party_players_ids) == 1 and party_players_ids[0] in team_players_ids:
                # Increment number of solo players if it belongs to a party with size 1
                num_solo_players += 1

        solo_players.append(num_solo_players)
    return solo_players


def get_num_parties(data, team):
    parties = []
    for parties_data, team_data in zip(data["parties"], data[team]):
        if not parties_data:
            parties.append(1)
            continue

        num_parties = 0
        team_players_ids = [p['id'] for p in team_data]

        for party_players_ids in parties_data.values():
            if party_players_ids[0] in team_players_ids:
                num_parties += 1

        parties.append(num_parties)
    return parties


def get_winner(scores):
    winners = []
    for score in scores:
        team_rounds = get_team_rounds(score)
        winners.append(0 if team_rounds[0] > team_rounds[1] else 1)
    return winners


def add_match_features(data):
    add_feature(data, get_mean_elo)
    add_feature(data, get_stddev_elo)
    add_feature(data, get_num_paid_memberships)
    add_feature(data, get_num_solo_players)
    add_feature(data, get_num_parties)

    data["match_mean_elo"] = (data["mean_elo_A"] + data["mean_elo_B"])/2

    data['entity_dummies_hub'] = np.where(data.entity == 'hub', 1, 0)
    data['entity_dummies_matchmaking'] = np.where(
        data.entity == 'matchmaking', 1, 0)

    data["5v5_free_queue"] = np.where(data.entityName == 'CS:GO 5v5', 1, 0)
    data["5v5_premium_queue"] = np.where(
        data.entityName == 'CS:GO 5v5 PREMIUM', 1, 0)
    data["winner"] = get_winner(data["score"])
