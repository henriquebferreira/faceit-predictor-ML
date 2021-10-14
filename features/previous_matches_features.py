from statistics import mean

import joblib

from features.utils import get_team_rounds

rating_predictor = joblib.load('./model_rating_predictor_2020_03_05_07_23_44_731552.pkl')

def has_won_the_match(match, player_id):
    players_team_A = [player['id'] for player in match['teamA']]
    is_on_teamA = True if player_id in players_team_A else False

    team_rounds = get_team_rounds(match['score'])
    return get_won_the_match(is_on_teamA, team_rounds)


def get_won_the_match(is_on_teamA, team_rounds):
    if is_on_teamA and team_rounds[0] > team_rounds[1]:
        won_the_match = True
    elif not is_on_teamA and team_rounds[1] > team_rounds[0]:
        won_the_match = True
    else:
        won_the_match = False
    return won_the_match


def get_mean_matches_on_map_prev(match, team, **kwargs):
    matches_on_map_ratio = []

    map_played = match['mapPlayed']
    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        same_map_series = player_prev_matches[player_prev_matches['mapPlayed'] == map_played]
        num_matches_same_map = same_map_series.shape[0]
        matches_on_map_ratio.append(num_matches_same_map)

        matches_on_map_ratio.append(num_matches_same_map)

    return mean(matches_on_map_ratio)


def get_mean_winrate_prev(match, team, **kwargs):
    players_winrate = []

    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        won_matches = player_prev_matches.apply(lambda m: has_won_the_match(m, player_id), axis=1)
        num_won_matches = won_matches[won_matches == True].shape[0]
        players_winrate.append(num_won_matches / 10)

    return mean(players_winrate)


def compute_multikills_score(row, player_id):
    player = [player for team in row['teams'] for player in team if player['id'] == player_id][0]
    if 'playerStats' in player:
        player_stats = player['playerStats']
    else:
        return 0
    multi_kills_score = player_stats['tripleKills'] * 9 + player_stats['quadraKills'] * 16 + player_stats[
        'pentaKills'] * 25
    return multi_kills_score


def get_multikills_score_prev(match, team, **kwargs):
    # Multi kills weights: 3K : 9, 4K: 16, 5k: 25
    all_multi_kills = []

    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        multi_kills_score = player_prev_matches.apply(compute_multikills_score, player_id=player_id, axis=1)
        all_multi_kills.append(mean(multi_kills_score))

    return mean(all_multi_kills)


def compute_assists(row, player_id):
    player = [player for team in row['teams'] for player in team if player['id'] == player_id][0]
    if 'playerStats' in player:
        player_stats = player['playerStats']
    else:
        return 0
    return player_stats['assists']


def get_mean_assists_prev(match, team, **kwargs):
    all_assists = []

    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        player_assists = player_prev_matches.apply(compute_assists, player_id=player_id, axis=1)
        all_assists.append(mean(player_assists))

    return mean(all_assists)


def compute_kd(row, player_id):
    player = [player for team in row['teams'] for player in team if player['id'] == player_id][0]
    if 'playerStats' in player:
        player_stats = player['playerStats']
    else:
        return 0
    kills = player_stats['kills']
    deaths = player_stats['deaths']
    if deaths == 0:
        deaths += 1
    kd = kills / deaths
    return kd


def get_mean_kd_prev(match, team, **kwargs):
    all_kds = []

    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        player_kds = player_prev_matches.apply(compute_kd, player_id=player_id, axis=1)
        all_kds.append(mean(player_kds))

    return mean(all_kds)


def get_num_rounds(match):
    score_string = match['score'].split("/")
    num_rounds = sum(map(lambda r: int(r), score_string))
    return num_rounds


def compute_rating(match, player_id):
    player = [player for team in match['teams'] for player in team if player['id'] == player_id][0]
    if 'playerStats' in player:
        player_stats = player['playerStats']
    else:
        return 0
    AVERAGE_KPR = 0.679  # average kills per round
    AVERAGE_SPR = 0.317  # average survived rounds per round
    AVERAGE_RMK = 1.277  # average value calculated from rounds with multiple kills

    AVERAGE_APR = 1 #
    AVERAGE_MPR = 1 #
    num_rounds = get_num_rounds(match)
    kills = player_stats['kills']
    deaths = player_stats['deaths']
    triple_kills = player_stats['tripleKills']
    quadra_kills = player_stats['quadraKills']
    penta_kills = player_stats['pentaKills']
    non_multi_kills = kills - triple_kills * 3 - quadra_kills * 4 - penta_kills * 5

    kill_rating = (kills / num_rounds) / AVERAGE_KPR
    survival_rating = ((num_rounds - deaths) / num_rounds) / AVERAGE_SPR
    multi_kills_rating = ((non_multi_kills * 2 +
                           triple_kills * 9 +
                           quadra_kills * 16 +
                           penta_kills * 25) / num_rounds) / AVERAGE_RMK

    return (kill_rating + 0.7 * survival_rating + multi_kills_rating) / 2.7


def compute_delta_rating(match, player_id):
    player_rating = compute_rating(match, player_id)
    players_team_A = [player['id'] for player in match['teamA']]
    is_on_team_A = player_id in players_team_A

    team_rounds = get_team_rounds(match['score'])
    if is_on_team_A:
        dif_rounds = team_rounds[0] - team_rounds[1]
        opposing_team_elos = [player['elo'] for player in match['teamB']]
        player_elo = next(player['elo'] for player in match['teamA'] if player['id'] == player_id)
        dif_elo = player_elo - mean(opposing_team_elos)
    else:
        dif_rounds = team_rounds[1] - team_rounds[0]
        opposing_team_elos = [player['elo'] for player in match['teamA']]
        player_elo = next(player['elo'] for player in match['teamB'] if player['id'] == player_id)
        dif_elo = player_elo - mean(opposing_team_elos)

    to_predict = [dif_elo, dif_rounds]
    predicted_rating = rating_predictor.predict([to_predict])
    return player_rating - predicted_rating[0]


def get_mean_rating_prev(match, team, **kwargs):
    all_ratings = []

    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        player_ratings = player_prev_matches.apply(compute_rating, player_id=player_id, axis=1)
        all_ratings.append(mean(player_ratings))

    return mean(all_ratings)


def get_mean_delta_rating_prev(match, team, **kwargs):
    all_delta_ratings = []

    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        player_delta_ratings = player_prev_matches.apply(compute_delta_rating, player_id=player_id, axis=1)
        all_delta_ratings.append(mean(player_delta_ratings))

    return mean(all_delta_ratings)


def compute_interval_time(row, start_time):
    return row['startTime'] - start_time


def get_mean_interval_time_prev(match, team, **kwargs):
    all_intervals_time = []

    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']
    start_time = match['startTime']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        player_intervals_time = player_prev_matches.apply(compute_interval_time, start_time=start_time, axis=1)
        all_intervals_time.append(mean(player_intervals_time))

    return mean(all_intervals_time)


def get_max_interval_time_prev(match, team, **kwargs):
    all_intervals_time = []

    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']
    start_time = match['startTime']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        player_intervals_time = player_prev_matches.apply(compute_interval_time, start_time=start_time, axis=1)
        all_intervals_time.append(max(player_intervals_time))

    return mean(all_intervals_time)


def compute_diff_to_mean_elo(row, player_id):
    elo = [player for team in row['teams'] for player in team if player['id'] == player_id][0]['elo']
    return elo


def get_mean_delta_elo_prev(match, team, **kwargs):
    all_elo_diff = []

    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        current_elo = [player for team in match['teams'] for player in team if player['id'] == player_id][0]['elo']
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        player_elos = player_prev_matches.apply(compute_diff_to_mean_elo, player_id=player_id, axis=1)

        mean_elo = mean(player_elos)
        all_elo_diff.append(current_elo - mean_elo)

    return mean(all_elo_diff)


def get_num_played_togthr_prev(match, team, **kwargs):
    all_matches = []
    played_together = 0
    team_players_ids = [p['id'] for p in match[team]]
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        all_matches.extend(player_prev_matches['_id'].values)

    my_dict = {i: all_matches.count(i) for i in all_matches}
    my_dict = {k: v for k, v in my_dict.items() if v > 1}
    for k, v in my_dict.items():
        played_together += v
    return played_together


# 7 hours
on_day_time = 7 * 3600


def get_mean_first_matches_on_day(match, team, **kwargs):
    first_matches = []

    team_players_ids = [p['id'] for p in match[team]]
    start_time = match['startTime']
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        is_first_match = 1
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        for _, row in player_prev_matches.iterrows():
            interval_time = start_time - row['startTime']
            if interval_time <= on_day_time:
                is_first_match = 0
        first_matches.append(is_first_match)

    return mean(first_matches)


def get_mean_matches_on_day(match, team, **kwargs):
    matches_today = []

    team_players_ids = [p['id'] for p in match[team]]
    start_time = match['startTime']
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        player_matches_today = 0
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        for _, row in player_prev_matches.iterrows():
            interval_time = start_time - row['startTime']
            if interval_time <= on_day_time:
                player_matches_today += 1
        matches_today.append(player_matches_today)

    return mean(matches_today)


def get_mean_played_map_on_day(match, team, **kwargs):
    matches_on_map_today = []

    team_players_ids = [p['id'] for p in match[team]]
    start_time = match['startTime']
    map_played = match['mapPlayed']
    previous_matches = kwargs['prevs']
    p2m = kwargs['p2m']

    for player_id in team_players_ids:
        played_map_today = 0
        player_prev_matches = previous_matches[previous_matches['_id'].isin(p2m[player_id])]
        for _, row in player_prev_matches.iterrows():
            interval_time = start_time - row['startTime']
            if (interval_time <= on_day_time) and (row['mapPlayed'] == map_played):
                played_map_today = 1
        matches_on_map_today.append(played_map_today)

    return mean(matches_on_map_today)
