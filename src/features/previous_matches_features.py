from pymongo import MongoClient
from collections import defaultdict
from src.db.config import read_config
from src.features.utils import compute_feature, get_team_rounds, get_player_won_the_match, add_feature, AVERAGE_RMK, add_feature, compute_rating

db_cfg = read_config("local.ingestorDB")
client = MongoClient(**db_cfg)
db = client['faceit_imported']
matches_coll = db['match']


def get_player_to_previous(match):
    all_previous_matches_ids = set()
    player_to_previous = {}
    for team in (match["teamA"], match["teamB"]):
        for player in team:
            player_to_previous[player["id"]] = player["previousMatches"]
            all_previous_matches_ids = all_previous_matches_ids.union(
                player["previousMatches"])

    all_previous_matches_ids = list(all_previous_matches_ids)
    matches_cursor = matches_coll.find(
        {"_id": {"$in": all_previous_matches_ids}})
    previous_matches = {m["_id"]: m for m in matches_cursor}

    for player_id, prev_matches_ids_of_player in player_to_previous.items():
        prev_matches_of_player = [match for match_id, match in previous_matches.items(
        ) if match_id in prev_matches_ids_of_player]
        player_to_previous[player_id] = prev_matches_of_player

    return player_to_previous


def get_mean_matches_on_map_prev(match, team, **kwargs):
    num_matches_same_map = 0

    player_to_previous = kwargs['ptp']
    map_played = match['mapPlayed']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        num_matches_same_map += len(
            [m for m in player_prev_matches if m['mapPlayed'] == map_played])

    return num_matches_same_map / 5


def get_mean_winrate_prev(match, team, **kwargs):
    winrates = 0

    player_to_previous = kwargs['ptp']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        num_won_prev_matches = sum([get_player_won_the_match(
            m, player["id"]) for m in player_prev_matches])
        num_prev_matches = len(player_prev_matches)
        winrates += num_won_prev_matches / \
            num_prev_matches if num_prev_matches != 0 else 0.5

    return winrates / 5


def get_mean_kd_prev(match, team, **kwargs):
    kds = 0

    player_to_previous = kwargs['ptp']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_id = player["id"]
        prev_match_kds = []
        for prev_match in player_prev_matches:
            player_prev = [p for team in prev_match['teams']
                           for p in team if p['id'] == player_id][0]
            if 'playerStats' not in player_prev:
                continue
            player_stats = player_prev['playerStats']
            kills = player_stats['kills']
            deaths = player_stats['deaths']
            if deaths == 0:
                deaths += 1
            kd_ratio = kills / deaths
            prev_match_kds.append(kd_ratio)
        kds += sum(prev_match_kds) / \
            len(prev_match_kds) if prev_match_kds else 1

    return kds / 5


def get_mean_weighted_kd_by_elo_prev(match, team, **kwargs):
    kds = 0

    player_to_previous = kwargs['ptp']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_id = player["id"]
        player_elo = player["elo"]
        prev_match_kds = []
        for prev_match in player_prev_matches:
            player_prev = [p for team in prev_match['teams']
                           for p in team if p['id'] == player_id][0]
            if 'playerStats' not in player_prev:
                continue
            player_stats = player_prev['playerStats']
            kills = player_stats['kills']
            deaths = player_stats['deaths']
            if deaths == 0:
                deaths += 1
            prev_match_kds.append(kills / deaths)
        kds += sum(prev_match_kds) * player_elo / len(prev_match_kds)

    return kds / 5


def get_mean_multikills_score_prev(match, team, **kwargs):
    all_multikills = 0

    player_to_previous = kwargs['ptp']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_id = player["id"]
        prev_match_multikills = []
        for prev_match in player_prev_matches:
            player_prev = [player for team in prev_match['teams']
                           for player in team if player['id'] == player_id][0]
            if 'playerStats' not in player_prev:
                continue
            player_stats = player_prev['playerStats']
            triple_k = player_stats['tripleKills']
            quadra_k = player_stats['quadraKills']
            penta_k = player_stats['pentaKills']

            rounds = sum(get_team_rounds(prev_match['score']))
            if not rounds:
                continue
            multikills_score = (triple_k * 9 + quadra_k *
                                16 + penta_k * 25) / rounds
            prev_match_multikills.append(multikills_score)
        all_multikills += sum(prev_match_multikills) / len(
            prev_match_multikills) if prev_match_multikills else AVERAGE_RMK

    return all_multikills / 5


def get_mean_rating_prev(match, team, **kwargs):
    all_ratings = 0

    player_to_previous = kwargs['ptp']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_id = player["id"]
        prev_match_ratings = []
        for prev_match in player_prev_matches:
            player_prev = [player for team in prev_match['teams']
                           for player in team if player['id'] == player_id][0]
            if 'playerStats' not in player_prev:
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

            rating = compute_rating(
                kills, deaths, triple_k, quadra_k, penta_k, assists, mvps, rounds)
            prev_match_ratings.append(rating)
        all_ratings += sum(prev_match_ratings) / \
            len(prev_match_ratings) if prev_match_ratings else 1

    return all_ratings / 5


def get_mean_interval_time_prev(match, team, **kwargs):
    interval_time_prev = 0

    player_to_previous = kwargs['ptp']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_intervals = [start_time - prev_match['startTime']
                            for prev_match in player_prev_matches]
        interval_time_prev += sum(player_intervals) / len(player_intervals)

    return interval_time_prev / 5


def get_mean_interval_time_oldest_prev(match, team, **kwargs):
    interval_time_prev = 0

    player_to_previous = kwargs['ptp']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_intervals = [start_time - prev_match['startTime']
                            for prev_match in player_prev_matches]
        interval_time_prev += max(player_intervals)

    return interval_time_prev / 5


def get_mean_interval_time_most_recent_prev(match, team, **kwargs):
    interval_time_prev = 0

    player_to_previous = kwargs['ptp']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_intervals = [start_time - prev_match['startTime']
                            for prev_match in player_prev_matches]
        interval_time_prev += min(player_intervals)

    return interval_time_prev / 5


def get_max_interval_time_most_recent_prev(match, team, **kwargs):
    interval_time_prev = []

    player_to_previous = kwargs['ptp']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_intervals = [start_time - prev_match['startTime']
                            for prev_match in player_prev_matches]
        most_recent = min(player_intervals)
        interval_time_prev.append(most_recent)

    return max(interval_time_prev)


def get_mean_delta_elo_prev(match, team, **kwargs):
    delta_elos = 0

    player_to_previous = kwargs['ptp']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_id = player["id"]
        player_elo = player["elo"]
        prev_delta_elos = []
        for prev_match in player_prev_matches:
            player_prev_elo = [p for team in prev_match['teams']
                               for p in team if p['id'] == player_id][0]["elo"]
            prev_delta_elos.append(player_elo - player_prev_elo)
        delta_elos += sum(prev_delta_elos) / len(prev_delta_elos)

    return delta_elos / 5


def get_mean_dif_rounds_prev(match, team, **kwargs):
    dif_rounds = 0

    player_to_previous = kwargs['ptp']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

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

        dif_rounds += sum(prev_dif_rounds) / len(prev_dif_rounds)

    return dif_rounds / 5


def get_mean_dif_elo_prev(match, team, **kwargs):
    dif_elo = 0

    player_to_previous = kwargs['ptp']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

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
        dif_elo += sum(player_dif_elo) / len(player_dif_elo)

    return dif_elo / 5


def get_mean_matches_afk(match, team, **kwargs):
    afks = 0

    player_to_previous = kwargs['ptp']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        player_id = player["id"]
        for prev_match in player_prev_matches:
            player_prev = [player for team in prev_match['teams']
                           for player in team if player['id'] == player_id][0]
            if not 'playerStats' in player_prev:
                afks += 1

    return afks / 5


def get_num_played_togthr_prev(match, team, **kwargs):
    all_played_together = 0

    player_to_previous = kwargs['ptp']

    players_in_match = defaultdict(list)
    for player in match[team]:
        for prev_match_id in player["previousMatches"]:
            players_in_match[prev_match_id].append(player["id"])

    # for all previous that have two or more common: check if all in the same team
    for match_id, player_ids in players_in_match.items():
        if len(player_ids) > 1:
            player_prev_matches = player_to_previous[player_ids[0]]
            prev_match = [
                m for m in player_prev_matches if m["_id"] == match_id][0]

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

    player_to_previous = kwargs['ptp']

    players_in_match = defaultdict(list)
    for player in match[team]:
        for prev_match_id in player["previousMatches"]:
            players_in_match[prev_match_id].append(player["id"])

    # for all previous that have two or more common: check if all in the same team
    for match_id, player_ids in players_in_match.items():
        if len(player_ids) > 1:
            player_prev_matches = player_to_previous[player_ids[0]]
            prev_match = [
                m for m in player_prev_matches if m["_id"] == match_id][0]

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
    first_match_on_day = 0

    player_to_previous = kwargs['ptp']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        most_recent = min([start_time - prev_match['startTime']
                           for prev_match in player_prev_matches])
        # if most recent match was played more than 7 hours ago then mark as the first match of the day
        if most_recent > on_day_time:
            first_match_on_day += 1

    return first_match_on_day / 5


def get_mean_matches_on_day(match, team, **kwargs):
    num_matches_on_day = 0

    player_to_previous = kwargs['ptp']
    start_time = match['startTime']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        intervals = [start_time - prev_match['startTime']
                     for prev_match in player_prev_matches]
        num_matches_on_day += len([i for i in intervals if i < on_day_time])

    return num_matches_on_day / 5


def get_mean_played_map_on_day(match, team, **kwargs):
    num_matches_on_day = 0

    player_to_previous = kwargs['ptp']
    start_time = match['startTime']
    map_played = match['mapPlayed']

    for player in match[team]:
        player_prev_matches = player_to_previous[player["id"]]

        intervals = [start_time - prev_match['startTime']
                     for prev_match in player_prev_matches if prev_match['mapPlayed'] == map_played]
        num_matches_on_day += len([i for i in intervals if i < on_day_time])

    return num_matches_on_day / 5


def add_previous_matches_features(data):
    features_data = defaultdict(list)

    def compute_previous_features(match):
        ptp = get_player_to_previous(match)
        compute_feature(match, features_data,
                        get_mean_matches_on_map_prev, ptp=ptp)
        compute_feature(match, features_data, get_mean_winrate_prev, ptp=ptp)
        compute_feature(match, features_data, get_mean_kd_prev,  ptp=ptp)
        compute_feature(match, features_data,
                        get_mean_weighted_kd_by_elo_prev, ptp=ptp)
        compute_feature(match, features_data,
                        get_mean_multikills_score_prev, ptp=ptp)
        compute_feature(match, features_data, get_mean_rating_prev, ptp=ptp)
        compute_feature(match, features_data,
                        get_mean_interval_time_prev, ptp=ptp)
        compute_feature(match, features_data,
                        get_mean_interval_time_oldest_prev, ptp=ptp)
        compute_feature(
            match, features_data, get_mean_interval_time_most_recent_prev, ptp=ptp)
        compute_feature(match, features_data,
                        get_max_interval_time_most_recent_prev, ptp=ptp)
        compute_feature(match, features_data, get_mean_delta_elo_prev, ptp=ptp)
        compute_feature(match, features_data,
                        get_mean_dif_rounds_prev, ptp=ptp)
        compute_feature(match, features_data, get_mean_dif_elo_prev, ptp=ptp)
        compute_feature(match, features_data, get_mean_matches_afk, ptp=ptp)
        compute_feature(match, features_data,
                        get_num_played_togthr_prev, ptp=ptp)
        compute_feature(match, features_data, get_winrate_togthr_prev, ptp=ptp)
        compute_feature(match, features_data,
                        get_mean_first_matches_on_day, ptp=ptp)
        compute_feature(match, features_data, get_mean_matches_on_day, ptp=ptp)
        compute_feature(match, features_data,
                        get_mean_played_map_on_day, ptp=ptp)

    data.apply(compute_previous_features, axis=1)

    for ft_name, ft_values in features_data.items():
        data[ft_name] = ft_values

        if ft_name.endswith("_B"):
            ft_name = ft_name[:-2]
            data["dif_" + ft_name] = data[ft_name + "_A"] - data[ft_name + "_B"]
