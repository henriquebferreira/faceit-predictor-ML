from src.features.utils import AVERAGE_RMK, add_feature, compute_rating, summation_by


def get_mean_matches(data, team):
    mean_matches = []
    for team_data in data[team]:
        matches = 0
        for p in team_data:
            map_stats = p["mapStats"]
            matches += summation_by(map_stats.values(), 'matches')
        mean_matches.append(matches/5)
    return mean_matches


def get_mean_winrate(data, team):
    mean_winrate = []
    for team_data in data[team]:
        player_winrate = 0
        for p in team_data:
            map_stats = p["mapStats"]
            matches = summation_by(map_stats.values(), 'matches')
            wins = summation_by(map_stats.values(), 'wins')
            player_winrate += (wins / matches) if matches != 0 else 0.5
        mean_winrate.append(player_winrate/5)
    return mean_winrate


def get_mean_kd(data, team):
    mean_kd = []
    for team_data in data[team]:
        player_kd = 0
        for p in team_data:
            map_stats = p["mapStats"]
            kills = summation_by(map_stats.values(), 'kills')
            deaths = summation_by(map_stats.values(), 'deaths')
            player_kd += (kills / deaths) if deaths != 0 else 1
        mean_kd.append(player_kd/5)
    return mean_kd


def get_mean_multikills_score(data, team):
    mean_multikills_score = []
    for team_data in data[team]:
        multikills_score = 0
        for p in team_data:
            map_stats = p["mapStats"]
            triple_k = summation_by(map_stats.values(), 'tripleKills')
            quadra_k = summation_by(map_stats.values(), 'quadraKills')
            penta_k = summation_by(map_stats.values(), 'pentaKills')
            rounds = summation_by(map_stats.values(), 'rounds')

            multikills = triple_k * 9 + quadra_k * 16 + penta_k * 25
            multikills_score += (multikills /
                                 rounds) if rounds != 0 else AVERAGE_RMK
        mean_multikills_score.append(multikills_score/5)
    return mean_multikills_score


def get_mean_rating(data, team):
    mean_rating = []
    for team_data in data[team]:
        rating = 0
        for p in team_data:
            map_stats = p["mapStats"]
            kills = summation_by(map_stats.values(), 'kills')
            deaths = summation_by(map_stats.values(), 'deaths')
            triple_k = summation_by(map_stats.values(), 'tripleKills')
            quadra_k = summation_by(map_stats.values(), 'quadraKills')
            penta_k = summation_by(map_stats.values(), 'pentaKills')
            assists = summation_by(map_stats.values(), 'assists')
            mvps = summation_by(map_stats.values(), 'mvps')
            rounds = summation_by(map_stats.values(), 'rounds')

            rating += compute_rating(kills, deaths, triple_k,
                                     quadra_k, penta_k, assists, mvps, rounds)

        mean_rating.append(rating/5)
    return mean_rating


def get_mean_matches_on_map(data, team):
    mean_matches = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        matches = 0
        for p in team_data:
            map_stats = p["mapStats"]
            map_played_data = map_stats.get(map_played, {})
            matches += map_played_data.get('matches', 0)
        mean_matches.append(matches/5)
    return mean_matches


def get_mean_winrate_on_map(data, team):
    mean_winrate = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        player_winrate = 0
        for p in team_data:
            map_stats = p["mapStats"]
            map_played_data = map_stats.get(map_played, {})

            matches = map_played_data.get('matches', 0)
            wins = map_played_data.get('wins', 0)
            player_winrate += (wins / matches) if matches != 0 else 0.5
        mean_winrate.append(player_winrate/5)
    return mean_winrate


def get_mean_kd_on_map(data, team):
    mean_kd = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        player_kd = 0
        for p in team_data:
            map_stats = p["mapStats"]
            map_played_data = map_stats.get(map_played, {})
            kills = map_played_data.get('kills', 0)
            deaths = map_played_data.get('deaths', 0)
            player_kd += (kills / deaths) if deaths != 0 else 1
        mean_kd.append(player_kd/5)
    return mean_kd


def get_mean_multikills_score_on_map(data, team):
    mean_multikills_score = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        multikills_score = 0
        for p in team_data:
            map_stats = p["mapStats"]

            map_played_data = map_stats.get(map_played, {})
            triple_k = map_played_data.get('tripleKills', 0)
            quadra_k = map_played_data.get('quadraKills', 0)
            penta_k = map_played_data.get('pentaKills', 0)
            rounds = map_played_data.get('rounds', 0)

            multikills = triple_k * 9 + quadra_k * 16 + penta_k * 25
            multikills_score += (multikills /
                                 rounds) if rounds != 0 else AVERAGE_RMK
        mean_multikills_score.append(multikills_score/5)
    return mean_multikills_score


def get_mean_rating_on_map(data, team):
    mean_rating = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        rating = 0
        for p in team_data:
            map_stats = p["mapStats"]

            map_played_data = map_stats.get(map_played, {})
            kills = map_played_data.get('kills', 0)
            deaths = map_played_data.get('deaths', 0)
            triple_k = map_played_data.get('tripleKills', 0)
            quadra_k = map_played_data.get('quadraKills', 0)
            penta_k = map_played_data.get('pentaKills', 0)
            assists = map_played_data.get('assists', 0)
            mvps = map_played_data.get('mvps', 0)
            rounds = map_played_data.get('rounds', 0)

            rating += compute_rating(kills, deaths, triple_k,
                                     quadra_k, penta_k, assists,
                                     mvps, rounds)

        mean_rating.append(rating/5)
    return mean_rating


def get_mean_matches_map_preference(data, team):
    mean_preference = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        matches_pref = 0
        for p in team_data:
            map_stats = p["mapStats"]
            map_played_data = map_stats.get(map_played, {})
            matches = summation_by(map_stats.values(), 'matches')
            matches_on_map = map_played_data.get('matches', 0)

            matches_pref += matches_on_map / matches if matches else 0.125
        mean_preference.append(matches_pref/5)
    return mean_preference


def get_mean_winrate_map_preference(data, team):
    mean_winrate = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        winrate_pref = 0
        for p in team_data:
            map_stats = p["mapStats"]

            map_played_data = map_stats.get(map_played, {})
            matches = summation_by(map_stats.values(), 'matches')
            wins = summation_by(map_stats.values(), 'wins')
            matches_on_map = map_played_data.get('matches', 0)
            wins_on_map = map_played_data.get('wins', 0)

            if matches and matches_on_map and wins:
                winrate_on_map = wins_on_map / matches_on_map
                winrate = wins / matches
                winrate_pref += winrate_on_map / winrate
            else:
                winrate_pref += 1

        mean_winrate.append(winrate_pref/5)
    return mean_winrate


def get_mean_kd_map_preference(data, team):
    mean_kd = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        kd_pref = 0
        for p in team_data:
            map_stats = p["mapStats"]

            map_played_data = map_stats.get(map_played, {})
            kills = summation_by(map_stats.values(), 'kills')
            deaths = summation_by(map_stats.values(), 'deaths')
            kills_on_map = map_played_data.get('kills', 0)
            deaths_on_map = map_played_data.get('deaths', 0)

            if deaths and deaths_on_map and kills:
                kd_on_map = kills_on_map / deaths_on_map
                kd = kills / deaths
                kd_pref += (kd_on_map/kd)
            else:
                kd_pref += 1

        mean_kd.append(kd_pref/5)
    return mean_kd


def get_mean_multikills_score_map_preference(data, team):
    mean_multikills_score = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        multikills_score_pref = 0
        for p in team_data:
            map_stats = p["mapStats"]

            map_played_data = map_stats.get(map_played, {})
            triple_k = summation_by(map_stats.values(), 'tripleKills')
            quadra_k = summation_by(map_stats.values(), 'quadraKills')
            penta_k = summation_by(map_stats.values(), 'pentaKills')
            rounds = summation_by(map_stats.values(), 'rounds')

            triple_k_on_map = map_played_data.get('tripleKills', 0)
            quadra_k_on_map = map_played_data.get('quadraKills', 0)
            penta_k_on_map = map_played_data.get('pentaKills', 0)
            rounds_on_map = map_played_data.get('rounds', 0)

            if rounds and rounds_on_map:
                multikills_score_on_map = (triple_k_on_map * 9 +
                                           quadra_k_on_map * 16 +
                                           penta_k_on_map * 25) / rounds_on_map
                multikills_score = (triple_k * 9 +
                                    quadra_k * 16 +
                                    penta_k * 25) / rounds
                if not multikills_score:
                    multikills_score_pref += 1
                else:
                    multikills_score_pref += (multikills_score_on_map /
                                              multikills_score)
            else:
                multikills_score_pref += 1

        mean_multikills_score.append(multikills_score_pref/5)
    return mean_multikills_score


def get_mean_rating_map_preference(data, team):
    mean_rating = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        rating_preference = 0
        for p in team_data:
            map_stats = p["mapStats"]

            map_played_data = map_stats.get(map_played, {})
            kills = summation_by(map_stats.values(), 'kills')
            deaths = summation_by(map_stats.values(), 'deaths')
            assists = summation_by(map_stats.values(), 'assists')
            mvps = summation_by(map_stats.values(), 'mvps')
            triple_k = summation_by(map_stats.values(), 'tripleKills')
            quadra_k = summation_by(map_stats.values(), 'quadraKills')
            penta_k = summation_by(map_stats.values(), 'pentaKills')
            rounds = summation_by(map_stats.values(), 'rounds')

            kills_on_map = map_played_data.get('kills', 0)
            deaths_on_map = map_played_data.get('deaths', 0)
            triple_k_on_map = map_played_data.get('tripleKills', 0)
            quadra_k_on_map = map_played_data.get('quadraKills', 0)
            penta_k_on_map = map_played_data.get('pentaKills', 0)
            assists_on_map = map_played_data.get('assists', 0)
            mvps_on_map = map_played_data.get('mvps', 0)
            rounds_on_map = map_played_data.get('rounds', 0)

            if rounds and rounds_on_map:
                rating_on_map = compute_rating(kills_on_map, deaths_on_map, triple_k_on_map,
                                               quadra_k_on_map, penta_k_on_map, assists_on_map,
                                               mvps_on_map, rounds_on_map)
                rating = compute_rating(kills, deaths, triple_k,
                                        quadra_k, penta_k, assists,
                                        mvps, rounds)
                rating_preference += (rating_on_map / rating)
            else:
                rating_preference += 1

        mean_rating.append(rating_preference/5)
    return mean_rating


def add_lifetime_features(data):
    add_feature(data, get_mean_matches)
    add_feature(data, get_mean_winrate)
    add_feature(data, get_mean_kd)
    add_feature(data, get_mean_multikills_score)
    add_feature(data, get_mean_rating)

    add_feature(data, get_mean_matches_on_map)
    add_feature(data, get_mean_winrate_on_map)
    add_feature(data, get_mean_kd_on_map)
    add_feature(data, get_mean_multikills_score_on_map)
    add_feature(data, get_mean_rating_on_map)

    add_feature(data, get_mean_matches_map_preference)
    add_feature(data, get_mean_winrate_map_preference)
    add_feature(data, get_mean_kd_map_preference)
    add_feature(data, get_mean_multikills_score_map_preference)
    add_feature(data, get_mean_rating_map_preference)
