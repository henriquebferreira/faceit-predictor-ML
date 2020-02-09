from statistics import mean
import numpy as np
import pandas as pd

previous = 20


def add_feature(feature, function, data):
    teams = ("A", "B")

    for team in teams:
        feature_team = feature + "_" + team
        data[feature_team] = data.apply(
            lambda row: function(row, "team" + team), axis=1
        ).values
    data["dif_" + feature] = data[feature + "_A"] - data[feature + "_B"]


def get_new_players(row, team):
    new_players = 0
    for player_id, player_info in row[team].items():
        if (player_info["lifetimeData"] is None) or (
            player_info["lifetimeData"]["onMapData"] is None
        ):
            new_players += 1
            continue
    return new_players


def get_mean_matches(row, team):
    matches = []
    for player_id, player_info in row[team].items():
        if player_info["lifetimeData"] is None:
            matches.append(0)
            continue
        lifetime_data = player_info["lifetimeData"]
        matches.append(int(lifetime_data["matches"]))
    return mean(matches)


def get_mean_matches_on_map(row, team):
    matches = []
    for player_id, player_info in row[team].items():
        if (player_info["lifetimeData"] is None) or (
            player_info["lifetimeData"]["onMapData"] is None
        ):
            matches.append(0)
            continue
        on_map_data = player_info["lifetimeData"]["onMapData"]
        matches.append(int(on_map_data["matches"]))
    return mean(matches)


def get_mean_winrate_on_map(row, team):
    win_rate = []
    for player_id, player_info in row[team].items():
        if (player_info["lifetimeData"] is None) or (
            player_info["lifetimeData"]["onMapData"] is None
        ):
            # TODO: When player has no matches played on the map
            # win_rate.append(0)
            continue
        on_map_data = player_info["lifetimeData"]["onMapData"]
        win_rate.append(float(on_map_data["winRate"]))
    if not win_rate:
        return 50
    return mean(win_rate)


def get_mean_kd_on_map(row, team):
    kds = []
    for player_id, player_info in row[team].items():
        if (player_info["lifetimeData"] is None) or (
            player_info["lifetimeData"]["onMapData"] is None
        ):
            # TODO: When player has no matches played on the map
            # kds.append(0)
            continue
        on_map_data = player_info["lifetimeData"]["onMapData"]
        kds.append(float(on_map_data["averageKD"]))
    if not kds:
        return 1
    return mean(kds)


def get_mean_matches_preference(row, team):
    matches = []
    for player_id, player_info in row[team].items():
        if (player_info["lifetimeData"] is None) or (
            player_info["lifetimeData"]["onMapData"] is None
        ):
            matches.append(0)
            continue
        on_map_data = player_info["lifetimeData"]["onMapData"]
        lifetime_data = player_info["lifetimeData"]
        matches.append(int(on_map_data["matches"]) / int(lifetime_data["matches"]))
    if not matches:
        return 0
    return mean(matches)


def get_mean_winrate_preference(row, team):
    win_rate = []
    for player_id, player_info in row[team].items():
        if (player_info["lifetimeData"] is None) or (
            player_info["lifetimeData"]["onMapData"] is None
        ):
            win_rate.append(0)
            continue
        on_map_data = player_info["lifetimeData"]["onMapData"]
        lifetime_data = player_info["lifetimeData"]
        win_rate.append(int(on_map_data["winRate"]) - int(lifetime_data["winRate"]))
    if not win_rate:
        return 0
    return mean(win_rate)


def get_mean_kd_preference(row, team):
    kds = []
    for player_id, player_info in row[team].items():
        if (player_info["lifetimeData"] is None) or (
            player_info["lifetimeData"]["onMapData"] is None
        ):
            kds.append(0)
            continue
        on_map_data = player_info["lifetimeData"]["onMapData"]
        lifetime_data = player_info["lifetimeData"]
        kds.append(float(on_map_data["averageKD"]) - float(lifetime_data["averageKD"]))
    if not kds:
        return 0
    return mean(kds)


def get_mean_matches_on_map_prev(row, team):
    map_played = row["mapPlayed"]
    matches_on_map_ratio = []
    for player_id, player_info in row[team].items():
        maps_array = player_info["recentMatchesData"]["map"][:previous]
        if len(maps_array) == 0:
            continue
        maps_on_map_array = [x for x in maps_array if x == map_played]
        matches_on_map_ratio.append(len(maps_on_map_array) / len(maps_array))

    return mean(matches_on_map_ratio)


def get_mean_elo(row, team):
    elos = []
    for player_id, player_info in row[team].items():
        elos.append(player_info["elo"])
    return mean(elos)


def get_stddev_deviation_elo(row, team):
    elos = []
    for player_id, player_info in row[team].items():
        elos.append(player_info["elo"])
    return np.std(elos)


def get_paid_memberships(row, team):
    memberships = 0
    for player_id, player_info in row[team].items():
        membership = player_info["membership"]
        if membership != "free":
            memberships += 1
    return memberships


def get_solo_players(row, team):
    num_solo_players = 0
    if (row["parties"] == []) or (row["parties"] is None):
        return 0
    if row["parties"] == {"$undefined": True}:
        return 0
    for keys, values in row["parties"].items():
        first_player_team = values[0]
        if first_player_team in row[team].keys():
            if len(values) == 1:
                num_solo_players += 1
    return num_solo_players


def get_num_parties(row, team):
    num_parties = 0
    if (row["parties"] == []) or (row["parties"] is None):
        return 1
    if row["parties"] == {"$undefined": True}:
        return 1
    for keys, values in row["parties"].items():
        first_player_team = values[0]
        if first_player_team in row[team].keys():
            num_parties += 1
    return num_parties


def get_mean_winrate_prev(row, team):
    players_winrate = []
    for player_id, player_info in row[team].items():
        winner_array = player_info["recentMatchesData"]["wasWinner"][:previous]
        winner_array = [int(x) for x in winner_array]
        if not winner_array:
            players_winrate.append(0.5)
            continue
        players_winrate.append(mean(winner_array))
    return mean(players_winrate)


def get_multikills_prev(row, team):
    # Multi kills weights: 3K : 9, 4K: 16, 5k: 25
    all_multi_kills = []

    for player_id, player_info in row[team].items():
        multi_kills_score = 0
        if player_info["recentMatchesData"] is None:
            all_multi_kills.append(0)
            continue
        if not player_info["recentMatchesData"]["tripleKills"]:
            all_multi_kills.append(0)
            continue

        num_matches = len(player_info["recentMatchesData"]["tripleKills"][:previous])
        triple_array = player_info["recentMatchesData"]["tripleKills"][:previous]
        quadra_array = player_info["recentMatchesData"]["quadraKills"][:previous]
        penta_array = player_info["recentMatchesData"]["pentaKills"][:previous]

        multi_kills_score += sum(map(lambda x: int(x) * 9, triple_array))
        multi_kills_score += sum(map(lambda x: int(x) * 16, quadra_array))
        multi_kills_score += sum(map(lambda x: int(x) * 25, penta_array))
        all_multi_kills.append(multi_kills_score / num_matches)

    return mean(all_multi_kills)


def get_mean_assists_prev(row, team):
    all_assists = []
    for player_id, player_info in row[team].items():
        if not player_info["recentMatchesData"]["assists"]:
            all_assists.append(0)
            continue
        assists_array = player_info["recentMatchesData"]["assists"][:previous]
        all_assists.append(
            sum(map(lambda x: int(x), assists_array))
            / len(player_info["recentMatchesData"]["assists"][:previous])
        )

    return mean(all_assists)


def get_mean_kd_prev(row, team):
    kds = []

    for player_id, player_info in row[team].items():
        if not player_info["recentMatchesData"]["kills"]:
            kds.append(0)
            continue
        kills_array = player_info["recentMatchesData"]["kills"][:previous]
        deaths_array = player_info["recentMatchesData"]["deaths"][:previous]
        all_kills = sum(map(lambda x: int(x), kills_array))
        all_deaths = sum(map(lambda x: int(x), deaths_array))
        kds.append(all_kills / all_deaths)

    return mean(kds)


def get_mean_time_created_at(row, team):
    date_match = row["unix_start_time"]
    all_created_at = []
    for player_id, player_info in row[team].items():
        if (player_info["createdAt"]) == {"$undefined": True}:
            continue
        created_at = int(player_info["createdAt"]) // 1000
        all_created_at.append(date_match - created_at)

    if not all_created_at:
        return 0
    return mean(all_created_at)


def get_stddev_time_created_at(row, team):
    date_match = row["unix_start_time"]
    all_created_at = []
    for player_id, player_info in row[team].items():
        if (player_info["createdAt"]) == {"$undefined": True}:
            continue
        created_at = int(player_info["createdAt"]) // 1000
        all_created_at.append(date_match - created_at)

    if not all_created_at:
        return 0
    return np.std(all_created_at)


def get_min_time_created_at(row, team):
    date_match = row["unix_start_time"]
    all_created_at = []
    for player_id, player_info in row[team].items():
        if (player_info["createdAt"]) == {"$undefined": True}:
            continue
        created_at = int(player_info["createdAt"]) // 1000
        all_created_at.append(date_match - created_at)

    if not all_created_at:
        return 0
    return min(all_created_at)


def get_mean_time_prev(row, team):
    # Five days
    five_days = 5 * 24 * 3600

    date_match = row["unix_start_time"]
    all_recent_matches = []
    for player_id, player_info in row[team].items():
        recent_matches = 0
        date_array = player_info["recentMatchesData"]["date"][:previous]
        match_dates = list(map(lambda x: int(x) // 1000, date_array))
        for m in match_dates:
            if date_match - m <= five_days:
                recent_matches += 1
        all_recent_matches.append(recent_matches)

    return mean(all_recent_matches)


def get_max_time_prev(row, team):
    date_match = row["unix_start_time"]
    max_time = 0
    for player_id, player_info in row[team].items():
        player_time = 0
        date_array = player_info["recentMatchesData"]["date"][:previous]
        match_dates = list(map(lambda x: int(x) // 1000, date_array))
        for m in match_dates:
            player_time += date_match - m
        if player_time > max_time:
            max_time = player_time

    return max_time


def get_delta_mean_elo_prev(row, team):
    delta_to_mean_elo = []
    for player_id, player_info in row[team].items():
        elos_array = player_info["recentMatchesData"]["elo"][:previous]
        elo = player_info["elo"]
        elos_array = [i for i in elos_array if i != {"$undefined": True}]
        elos_array = [i for i in elos_array if i]
        if not elos_array:
            delta_to_mean_elo.append(0)
            continue
        mean_elo = mean(map(lambda x: int(x), elos_array))
        delta_to_mean_elo.append(elo - mean_elo)

    return mean(delta_to_mean_elo)


def get_first_match(row, team):
    # 7 hours
    seven_hours = 7 * 3600

    date_match = row["unix_start_time"]
    first_matches = 0
    for player_id, player_info in row[team].items():
        if not player_info["recentMatchesData"]["date"]:
            first_matches += 1
            continue
        date_most_recent = int(player_info["recentMatchesData"]["date"][0]) // 1000
        if date_match - date_most_recent > seven_hours:
            first_matches += 1

    return first_matches


def get_mean_matches_today(row, team):
    # 7 hours
    seven_hours = 7 * 3600

    date_match = row["unix_start_time"]
    all_num_matches_today = []
    for player_id, player_info in row[team].items():
        if not player_info["recentMatchesData"]["date"]:
            continue
        date_array = player_info["recentMatchesData"]["date"][:previous]
        match_dates = list(map(lambda x: int(x) // 1000, date_array))
        matches_today = list(
            filter(lambda x: date_match - x <= seven_hours, match_dates)
        )
        num_matches_today = len(matches_today)
        all_num_matches_today.append(num_matches_today)

    return mean(all_num_matches_today)


def get_played_map_today(row, team):
    # 7 hours
    seven_hours = 7 * 3600

    map_played = row["mapPlayed"]
    date_match = row["unix_start_time"]
    all_map_played = []
    for player_id, player_info in row[team].items():
        if not player_info["recentMatchesData"]["date"]:
            continue
        date_array = player_info["recentMatchesData"]["date"][:previous]
        map_array = player_info["recentMatchesData"]["map"][:previous]

        match_dates = list(map(lambda x: int(x) // 1000, date_array))
        date_map_dict = dict(zip(match_dates, map_array))
        date_map_dict = {
            k: v for k, v in date_map_dict.items() if date_match - k <= seven_hours
        }

        has_played_map_today = 1 if map_played in date_map_dict.values() else 0
        all_map_played.append(has_played_map_today)

    return sum(all_map_played)


def get_have_played_together_prev(row, team):
    all_matches = []
    played_together = 0
    for player_id, player_info in row[team].items():
        if not player_info["recentMatchesData"]["date"]:
            continue
        matches_array = player_info["recentMatchesData"]["matchId"][:previous]
        all_matches.extend(matches_array)
    my_dict = {i: all_matches.count(i) for i in all_matches}
    my_dict = {k: v for k, v in my_dict.items() if v > 1}
    for k, v in my_dict.items():
        played_together += v
    return played_together


def get_missing_info(row):
    for player_id, player_info in row["teamA"].items():
        if (player_info["lifetimeData"] is None) or (
            player_info["lifetimeData"]["onMapData"] is None
        ):
            return 1
        if len(player_info["recentMatchesData"]["wasWinner"][:previous]) != 20:
            return 1
    for player_id, player_info in row["teamB"].items():
        if (player_info["lifetimeData"] is None) or (
            player_info["lifetimeData"]["onMapData"] is None
        ):
            return 1
        if len(player_info["recentMatchesData"]["wasWinner"][:previous]) != 20:
            return 1
    return 0


def add_all_team_features(dataset):

    # Alternatively feature name could be programmed as:
    # function.__name__.split("_")[1:][0] returns (everything beyond get_)

    add_feature('new_players', get_new_players, dataset)
    add_feature('mean_matches', get_mean_matches, dataset)
    add_feature('mean_matches_on_map', get_mean_matches_on_map, dataset)
    add_feature('mean_winrate_on_map', get_mean_winrate_on_map, dataset)
    add_feature('mean_kd_on_map', get_mean_kd_on_map, dataset)
    add_feature('mean_matches_preference', get_mean_matches_preference, dataset)
    add_feature('mean_winrate_preference', get_mean_winrate_preference, dataset)
    add_feature('mean_kd_preference', get_mean_kd_preference, dataset)
    add_feature('mean_matches_on_map_prev', get_mean_matches_on_map_prev, dataset)
    add_feature('mean_elo', get_mean_elo, dataset)
    add_feature('stddev_elo', get_stddev_deviation_elo, dataset)
    add_feature('paid_memberships', get_paid_memberships, dataset)
    add_feature('solo_players', get_solo_players, dataset)
    add_feature('num_parties', get_num_parties, dataset)
    add_feature('mean_winrate_prev', get_mean_winrate_prev, dataset)
    add_feature('multikills_prev', get_multikills_prev, dataset)
    add_feature('mean_assists_prev', get_mean_assists_prev, dataset)
    add_feature('mean_kd_prev', get_mean_kd_prev, dataset)
    add_feature('mean_time_created_at', get_mean_time_created_at, dataset)
    add_feature('stddev_time_created_at', get_stddev_time_created_at, dataset)
    add_feature('min_time_created_at', get_min_time_created_at, dataset)
    add_feature('mean_time_prev', get_mean_time_prev, dataset)
    add_feature('max_time_prev', get_max_time_prev, dataset)
    add_feature('delta_mean_elo_prev', get_delta_mean_elo_prev, dataset)
    add_feature('first_match', get_first_match, dataset)
    add_feature('mean_matches_today', get_mean_matches_today, dataset)
    add_feature('played_map_today', get_played_map_today, dataset)
    add_feature('have_played_together_prev', get_have_played_together_prev, dataset)

    dataset.loc[:, 'missing_info'] = dataset.apply(lambda row: get_missing_info(row), axis=1).values

def add_all_match_features(dataset):
    date_format = "%Y-%m-%dT%H:%M:%SZ"

    dataset.loc[:, 'unix_start_time'] = pd.to_datetime(dataset['startTime'], format=date_format).values.astype(
        'datetime64[s]').astype('int')
    dataset.drop(columns=['startTime'], inplace=True)

    # TODO: redo concat below to be an inplace function
    dataset = pd.concat([dataset, pd.get_dummies(dataset['mapPlayed'], prefix='map')], axis=1)
    return dataset








