from statistics import mean

import joblib
import json
from pymongo import MongoClient
from collections import defaultdict
from src.db.config import read_config
from src.features.utils import get_team_rounds, get_player_won_the_match, add_feature

# rating_predictor = joblib.load(
#     './model_rating_predictor_2020_03_05_07_23_44_731552.pkl')

# Load performance statistics that were previously computed
with open('../data/external/performance_statistics.json') as fp:
    performance_statistics = json.load(fp)

AVERAGE_KPR = performance_statistics["meanKPR"]
AVERAGE_SPR = performance_statistics["meanSPR"]
AVERAGE_RMK = performance_statistics["meanMKPR"]
AVERAGE_APR = performance_statistics["meanAPR"]
AVERAGE_MVPPR = performance_statistics["meanMVPPR"]

db_cfg = read_config("local.ingestorDB")
client = MongoClient(**db_cfg)
db = client['faceit_imported']
# Connect to the collections inside the local ingestor database
players_coll = db['player']
matches_coll = db['match']
lifetime_stats_coll = db['player_lifetime_stats']


def get_all_previous_matches(match):
    previous_matches_ids = set()

    for team in (match["teamA"], match["teamB"]):
        for player in team:
            previous_matches_ids = previous_matches_ids.union(
                player["previousMatches"])

    previous_matches_cursor = matches_coll.find(
        {"_id": {"$in": list(previous_matches_ids)}})
    return {m["_id"]: m for m in previous_matches_cursor}


def get_mean_matches_on_map_prev(match, team, **kwargs):
    num_matches_same_map = 0

    previous_matches = kwargs['previous_matches']
    map_played = match['mapPlayed']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        num_matches_same_map += len(
            [m for m in player_prev_matches if m['mapPlayed'] == map_played])

    return num_matches_same_map / 5


def get_mean_winrate_prev(match, team, **kwargs):
    winrates = 0

    previous_matches = kwargs['previous_matches']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        num_won_prev_matches = sum([get_player_won_the_match(
            m, player["id"]) for m in player_prev_matches])
        num_prev_matches = len(player_prev_matches)
        winrates += num_won_prev_matches / \
            num_prev_matches if num_prev_matches != 0 else 0.5

    return winrates / 5


def get_mean_kd_prev(match, team, **kwargs):
    kds = 0

    previous_matches = kwargs['previous_matches']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_id = player["id"]
        prev_match_kds = []
        for prev_match in player_prev_matches:
            player_prev = [p for team in prev_match['teams']
                           for p in team if p['id'] == player_id][0]
            if 'playerStats' not in player_prev:
                prev_match_kds.append(1)
                continue
            player_stats = player_prev['playerStats']
            kills = player_stats['kills']
            deaths = player_stats['deaths']
            kd_ratio = (kills / deaths) if deaths != 0 else kills
            prev_match_kds.append(kd_ratio)
        kds += sum(prev_match_kds) / \
            len(prev_match_kds) if prev_match_kds else 1

    return kds / 5


def get_mean_weighted_kd_by_elo_prev(match, team, **kwargs):
    kds = 0

    previous_matches = kwargs['previous_matches']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_id = player["id"]
        player_elo = player["elo"]
        prev_match_kds = []
        for prev_match in player_prev_matches:
            player_prev = [p for team in prev_match['teams']
                           for p in team if p['id'] == player_id][0]
            if 'playerStats' not in player_prev:
                prev_match_kds.append(1)
                continue
            player_stats = player_prev['playerStats']
            kills = player_stats['kills']
            deaths = player_stats['deaths']
            kd_ratio = (kills / deaths) if deaths != 0 else kills
            prev_match_kds.append(kd_ratio)
        kds += sum(prev_match_kds) * player_elo / \
            len(prev_match_kds) if prev_match_kds else 1

    return kds / 5


def get_multikills_score_prev(match, team, **kwargs):
    all_multikills = 0

    previous_matches = kwargs['previous_matches']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_id = player["id"]
        prev_match_multikills = []
        for prev_match in player_prev_matches:
            player_prev = [player for team in prev_match['teams']
                           for player in team if player['id'] == player_id][0]
            if 'playerStats' not in player_prev:
                prev_match_multikills.append(AVERAGE_RMK)
                continue
            player_stats = player_prev['playerStats']
            triple_k = player_stats['tripleKills']
            quadra_k = player_stats['quadraKills']
            penta_k = player_stats['pentaKills']

            rounds = sum(get_team_rounds(prev_match['score']))
            multikills_score = (triple_k * 9 + quadra_k * 16 +
                                penta_k * 25) / rounds if rounds else AVERAGE_RMK
            prev_match_multikills.append(multikills_score)
        all_multikills += sum(prev_match_multikills) / len(
            prev_match_multikills) if prev_match_multikills else AVERAGE_RMK

    return all_multikills / 5


def get_mean_rating_prev(match, team, **kwargs):
    all_ratings = 0

    previous_matches = kwargs['previous_matches']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_id = player["id"]
        prev_match_ratings = []
        for prev_match in player_prev_matches:
            player_prev = [player for team in prev_match['teams']
                           for player in team if player['id'] == player_id][0]
            if 'playerStats' not in player_prev:
                prev_match_ratings.append(1)
                continue
            player_stats = player_prev['playerStats']
            kills = player_stats['kills']
            deaths = player_stats['deaths']
            triple_k = player_stats['tripleKills']
            quadra_k = player_stats['quadraKills']
            penta_k = player_stats['pentaKills']
            assists = player_stats['assists']
            mvps = player_stats['mvps']
            rounds = sum(get_team_rounds(prev_match['score']))

            kill_rating = kills / rounds / AVERAGE_KPR
            survival_rating = (rounds - deaths) / rounds / AVERAGE_SPR
            multi_kills_score = triple_k * 9 + quadra_k * 16 + penta_k * 25
            multi_kills_rating = multi_kills_score / rounds / AVERAGE_RMK
            assists_rating = assists / rounds / AVERAGE_APR
            mvps_rating = mvps / rounds / AVERAGE_MVPPR
            prev_match_ratings.append((kill_rating + 0.7 * survival_rating
                                       + multi_kills_rating + 0.5 * assists_rating
                                       + 0.3 * mvps_rating) / 3.5)
        all_ratings += sum(prev_match_ratings) / \
            len(prev_match_ratings) if prev_match_ratings else 1

    return all_ratings / 5


def get_mean_interval_time_prev(match, team, **kwargs):
    interval_time_prev = 0

    previous_matches = kwargs['previous_matches']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_intervals = [start_time - prev_match['startTime']
                            for prev_match in player_prev_matches]
        interval_time_prev += sum(player_intervals) / \
            len(player_intervals) if player_intervals else 0

    return interval_time_prev / 5


def get_mean_interval_time_oldest_prev(match, team, **kwargs):
    interval_time_prev = 0

    previous_matches = kwargs['previous_matches']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_intervals = [start_time - prev_match['startTime']
                            for prev_match in player_prev_matches]
        interval_time_prev += max(player_intervals) if player_intervals else 0

    return interval_time_prev / 5


def get_mean_interval_time_most_recent_prev(match, team, **kwargs):
    interval_time_prev = 0

    previous_matches = kwargs['previous_matches']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_intervals = [start_time - prev_match['startTime']
                            for prev_match in player_prev_matches]
        interval_time_prev += min(player_intervals) if player_intervals else 0

    return interval_time_prev / 5


def get_max_interval_time_most_recent_prev(match, team, **kwargs):
    interval_time_prev = []

    previous_matches = kwargs['previous_matches']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_intervals = [start_time - prev_match['startTime']
                            for prev_match in player_prev_matches]
        most_recent = min(player_intervals) if player_intervals else 0
        interval_time_prev.append(most_recent)

    return max(interval_time_prev) if interval_time_prev else 0


def get_mean_delta_elo_prev(match, team, **kwargs):
    delta_elos = 0

    previous_matches = kwargs['previous_matches']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_id = player["id"]
        player_elo = player["elo"]
        prev_delta_elos = []
        for prev_match in player_prev_matches:
            player_prev_elo = [p for team in prev_match['teams']
                               for p in team if p['id'] == player_id][0]["elo"]
            prev_delta_elos.append(player_elo - player_prev_elo)
        delta_elos += sum(prev_delta_elos) / \
            len(prev_delta_elos) if prev_delta_elos else 0

    return delta_elos / 5


def get_mean_dif_rounds_prev(match, team, **kwargs):
    dif_rounds = 0

    previous_matches = kwargs['previous_matches']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_id = player["id"]
        prev_dif_rounds = []
        for prev_match in player_prev_matches:
            is_on_team_A = any(
                [player['id'] == player_id for p in prev_match['teamA']])
            team_rounds = get_team_rounds(prev_match['score'])
            dif_team_rounds = team_rounds[0] - \
                team_rounds[1] if is_on_team_A else team_rounds[1] - \
                team_rounds[0]
            prev_dif_rounds.append(dif_team_rounds)

        dif_rounds += sum(prev_dif_rounds) / \
            len(prev_dif_rounds) if prev_dif_rounds else 0

    return dif_rounds / 5


def get_mean_dif_elo_prev(match, team, **kwargs):
    dif_elo = 0

    previous_matches = kwargs['previous_matches']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_id = player["id"]
        player_dif_elo = []
        for prev_match in player_prev_matches:
            is_on_team_A = any(
                [player['id'] == player_id for p in prev_match['teamA']])
            player_elo = [player for team in prev_match['teams']
                          for player in team if player['id'] == player_id][0]['elo']
            if is_on_team_A:
                elos_opposing_team = [player['elo']
                                      for player in prev_match['teamB']]
            else:
                elos_opposing_team = [player['elo']
                                      for player in prev_match['teamA']]

            mean_elo_opposing_team = sum(
                elos_opposing_team) / len(elos_opposing_team)
            player_dif_elo.append(player_elo - mean_elo_opposing_team)
        dif_elo += sum(player_dif_elo) / \
            len(player_dif_elo) if player_dif_elo else 0

    return dif_elo / 5


def get_mean_matches_afk(match, team, **kwargs):
    afks = 0

    previous_matches = kwargs['previous_matches']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        player_id = player["id"]
        prev_match_afks = 0
        for prev_match in player_prev_matches:
            player_prev = [player for team in prev_match['teams']
                           for player in team if player['id'] == player_id][0]
            if not 'playerStats' in player_prev:
                prev_match_afks += 1

        afks += prev_match_afks / \
            len(player_prev_matches) if player_prev_matches else 0

    return afks / 5


def get_num_played_togthr_prev(match, team, **kwargs):
    all_played_together = 0

    previous_matches = kwargs['previous_matches']
    team_players_ids = [p['id'] for p in match[team]]

    players_in_match = defaultdict(list)
    for player in match[team]:
        for prev_match_id in player["previousMatches"]:
            players_in_match[prev_match_id].append(player["id"])

    # for all previous that have two or more common: check if all in the same team
    for match_id, player_ids in players_in_match.items():
        if len(player_ids) > 1:
            prev_match = previous_matches[match_id]

            players_ids_A = [p['id'] for p in prev_match["teamA"]]
            players_ids_B = [p['id'] for p in prev_match["teamB"]]
            players_on_A = [p for p in player_ids if p in players_ids_A]
            players_on_B = [p for p in player_ids if p in players_ids_B]
            if len(players_on_A) > 1:
                all_played_together += len(players_on_A)
            if len(players_on_B) > 1:
                all_played_together += len(players_on_B)

    num_matches = sum([len(p) for p in players_in_match.values()])
    return all_played_together / num_matches


def get_winrate_togthr_prev(match, team, **kwargs):
    wins_together, num_matches_together = 0, 0

    previous_matches = kwargs['previous_matches']
    team_players_ids = [p['id'] for p in match[team]]

    players_in_match = defaultdict(list)
    for player in match[team]:
        for prev_match_id in player["previousMatches"]:
            players_in_match[prev_match_id].append(player["id"])

    # for all previous that have two or more common: check if all in the same team
    for match_id, player_ids in players_in_match.items():
        if len(player_ids) > 1:
            prev_match = previous_matches[match_id]

            players_ids_A = [p['id'] for p in prev_match["teamA"]]
            players_ids_B = [p['id'] for p in prev_match["teamB"]]
            players_on_A = [p for p in player_ids if p in players_ids_A]
            players_on_B = [p for p in player_ids if p in players_ids_B]
            if len(players_on_A) > 1:
                won_match = get_player_won_the_match(
                    prev_match, players_on_A[0])
                won_multiplier = 1 if won_match == 1 else -1
                wins_together += won_multiplier * len(players_on_A)
                num_matches_together += len(players_on_A)
            if len(players_on_B) > 1:
                won_match = get_player_won_the_match(
                    prev_match, players_on_B[0])
                won_multiplier = 1 if won_match == 1 else -1
                wins_together += won_multiplier * len(players_on_B)
                num_matches_together += len(players_on_B)

    return wins_together / num_matches_together if num_matches_together else 0


# 7 hours
on_day_time = 7 * 3600


def get_mean_first_matches_on_day(match, team, **kwargs):
    most_recent_matches_intervals = []

    previous_matches = kwargs['previous_matches']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        most_recent = min([start_time - prev_match['startTime']
                           for prev_match in player_prev_matches])
        most_recent_matches_intervals.append(most_recent)

    # if most recent match was played more than 7 hours ago then mark as the first match of the day
    return len([i for i in most_recent_matches_intervals if i > on_day_time])


def get_mean_matches_on_day(match, team, **kwargs):
    num_matches_on_day = 0

    previous_matches = kwargs['previous_matches']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        intervals = [start_time - prev_match['startTime']
                     for prev_match in player_prev_matches]
        num_matches_on_day += len([i for i in intervals if i < on_day_time])

    return num_matches_on_day / 5


def get_mean_played_map_on_day(match, team, **kwargs):
    num_matches_on_day = 0

    previous_matches = kwargs['previous_matches']
    start_time = match['startTime']
    map_played = match['mapPlayed']

    for player in match[team]:
        player_prev_matches_ids = player["previousMatches"]
        player_prev_matches = [m for match_id, m in previous_matches.items(
        ) if match_id in player_prev_matches_ids]

        intervals = [start_time - prev_match['startTime']
                     for prev_match in player_prev_matches if prev_match['mapPlayed'] == map_played]
        num_matches_on_day += len([i for i in intervals if i < on_day_time])

    return num_matches_on_day / 5


def add_previous_matches_features(match):
    previous_matches = get_all_previous_matches(match)

    add_feature(match, get_mean_matches_on_map_prev,
                previous_matches=previous_matches)
    add_feature(match, get_mean_winrate_prev,
                previous_matches=previous_matches)
    add_feature(match, get_mean_kd_prev,  previous_matches=previous_matches)
    add_feature(match, get_mean_weighted_kd_by_elo_prev,
                previous_matches=previous_matches)

    add_feature(match, get_multikills_score_prev,
                previous_matches=previous_matches)
    add_feature(match, get_mean_rating_prev, previous_matches=previous_matches)

    # # add_feature(match, get_mean_delta_rounds_predictor_prev,  previous_matches=previous_matches)
    # # ##    add_feature(match, get_mean_delta_rating_prev, prevs = previous_matches, p2m=players_to_match)
    add_feature(match, get_mean_interval_time_prev,
                previous_matches=previous_matches)
    add_feature(match, get_mean_interval_time_oldest_prev,
                previous_matches=previous_matches)

    add_feature(match, get_mean_interval_time_most_recent_prev,
                previous_matches=previous_matches)
    add_feature(match, get_max_interval_time_most_recent_prev,
                previous_matches=previous_matches)

    add_feature(match, get_mean_delta_elo_prev,
                previous_matches=previous_matches)
    add_feature(match, get_mean_dif_rounds_prev,
                previous_matches=previous_matches)
    add_feature(match, get_mean_dif_elo_prev,
                previous_matches=previous_matches)

    add_feature(match, get_mean_matches_afk, previous_matches=previous_matches)

    add_feature(match, get_num_played_togthr_prev,
                previous_matches=previous_matches)
    add_feature(match, get_winrate_togthr_prev,
                previous_matches=previous_matches)

    add_feature(match, get_mean_first_matches_on_day,
                previous_matches=previous_matches)
    add_feature(match, get_mean_matches_on_day,
                previous_matches=previous_matches)
    add_feature(match, get_mean_played_map_on_day,
                previous_matches=previous_matches)

    return match
