from src.features.utils import add_feature
import json

# Load performance statistics that were previously computed
with open('../data/external/performance_statistics.json') as fp:
    performance_statistics = json.load(fp)

AVERAGE_KPR = performance_statistics["meanKPR"]
AVERAGE_SPR = performance_statistics["meanSPR"]
AVERAGE_RMK = performance_statistics["meanMKPR"]
AVERAGE_APR = performance_statistics["meanAPR"]
AVERAGE_MVPPR = performance_statistics["meanMVPPR"]


def get_mean_matches(data, team):
    mean_matches = []
    for team_data in data[team]:
        total_matches = 0
        for p in team_data:
            map_stats = p["mapStats"]
            total_matches += sum([ms['matches'] for ms in map_stats.values()])
        mean_matches.append(total_matches/5)
    return mean_matches


def get_mean_winrate(data, team):
    mean_winrate = []
    for team_data in data[team]:
        player_winrate = 0
        for p in team_data:
            total_matches, total_wins = 0, 0
            map_stats = p["mapStats"]
            total_matches += sum([ms['matches'] for ms in map_stats.values()])
            total_wins += sum([ms['wins'] for ms in map_stats.values()])
            player_winrate += (total_wins /
                               total_matches) if total_matches != 0 else 0.5
        mean_winrate.append(player_winrate/5)
    return mean_winrate


def get_mean_kd(data, team):
    mean_kd = []
    for team_data in data[team]:
        player_kd = 0
        for p in team_data:
            total_kills, total_deaths = 0, 0
            map_stats = p["mapStats"]
            total_kills += sum([ms['kills'] for ms in map_stats.values()])
            total_deaths += sum([ms['deaths'] for ms in map_stats.values()])
            player_kd += (total_kills /
                          total_deaths) if total_deaths != 0 else 1
        mean_kd.append(player_kd/5)
    return mean_kd


def get_mean_multikills_score(data, team):
    mean_multikills_score = []
    for team_data in data[team]:
        multikills_score = 0
        for p in team_data:
            total_triple_k, total_quadra_k, total_penta_k, total_rounds = 0, 0, 0, 0
            map_stats = p["mapStats"]

            total_triple_k += sum([ms['tripleKills']
                                   for ms in map_stats.values()])
            total_quadra_k += sum([ms['quadraKills']
                                   for ms in map_stats.values()])
            total_penta_k += sum([ms['pentaKills']
                                  for ms in map_stats.values()])
            total_rounds += sum([ms['rounds'] for ms in map_stats.values()])

            multikills = total_triple_k * 9 + total_quadra_k * 16 + total_penta_k * 25
            multikills_score += (multikills /
                                 total_rounds) if total_rounds != 0 else AVERAGE_RMK
        mean_multikills_score.append(multikills_score/5)
    return mean_multikills_score


def compute_rating(kills, deaths, triple_k, quadra_k, penta_k, assists, mvps, rounds):
    kill_rating = kills / rounds / AVERAGE_KPR
    survival_rating = (rounds - deaths) / rounds / AVERAGE_SPR
    multi_kills_score = triple_k * 9 + quadra_k * 16 + penta_k * 25
    multi_kills_rating = multi_kills_score / rounds / AVERAGE_RMK
    assists_rating = assists / rounds / AVERAGE_APR
    mvps_rating = mvps / rounds / AVERAGE_MVPPR

    rating = (kill_rating + 0.7 * survival_rating
              + multi_kills_rating
              + 0.5 * assists_rating
              + 0.3 * mvps_rating) / 3.5
    return rating


def get_mean_rating(data, team):
    mean_rating = []
    for team_data in data[team]:
        rating = 0
        for p in team_data:
            total_kills, total_deaths, total_assists, total_mvps = 0, 0, 0, 0
            total_triple_k, total_quadra_k, total_penta_k, total_rounds = 0, 0, 0, 0
            map_stats = p["mapStats"]

            total_kills += sum([ms['kills'] for ms in map_stats.values()])
            total_deaths += sum([ms['deaths'] for ms in map_stats.values()])
            total_assists += sum([ms['assists'] for ms in map_stats.values()])
            total_mvps += sum([ms['mvps'] for ms in map_stats.values()])
            total_triple_k += sum([ms['tripleKills']
                                   for ms in map_stats.values()])
            total_quadra_k += sum([ms['quadraKills']
                                   for ms in map_stats.values()])
            total_penta_k += sum([ms['pentaKills']
                                  for ms in map_stats.values()])
            total_rounds += sum([ms['rounds'] for ms in map_stats.values()])

            if total_rounds == 0:
                # TODO: review better value... impute mean
                rating += 1
            else:
                rating += compute_rating(total_kills, total_deaths, total_triple_k,
                                         total_quadra_k, total_penta_k, total_assists,
                                         total_mvps, total_rounds)

        mean_rating.append(rating/5)
    return mean_rating


def get_mean_matches_on_map(data, team):
    mean_matches = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        total_matches = 0
        for p in team_data:
            map_stats = p["mapStats"]
            total_matches += sum([ms['matches']
                                  for ms in map_stats.values() if ms["name"] == map_played])
        mean_matches.append(total_matches/5)
    return mean_matches


def get_mean_winrate_on_map(data, team):
    mean_winrate = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        player_winrate = 0
        for p in team_data:
            map_stats = p["mapStats"]
            total_matches = sum(
                [ms['matches'] for ms in map_stats.values() if ms["name"] == map_played])
            total_wins = sum(
                [ms['wins'] for ms in map_stats.values() if ms["name"] == map_played])
            player_winrate += (total_wins /
                               total_matches) if total_matches != 0 else 0.5
        mean_winrate.append(player_winrate/5)
    return mean_winrate


def get_mean_kd_on_map(data, team):
    mean_kd = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        player_kd = 0
        for p in team_data:
            map_stats = p["mapStats"]
            total_kills = sum(
                [ms['kills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_deaths = sum(
                [ms['deaths'] for ms in map_stats.values() if ms["name"] == map_played])
            player_kd += (total_kills /
                          total_deaths) if total_deaths != 0 else 1
        mean_kd.append(player_kd/5)
    return mean_kd


def get_mean_multikills_score_on_map(data, team):
    mean_multikills_score = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        multikills_score = 0
        for p in team_data:
            map_stats = p["mapStats"]

            total_triple_k = sum(
                [ms['tripleKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_quadra_k = sum(
                [ms['quadraKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_penta_k = sum(
                [ms['pentaKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_rounds = sum(
                [ms['rounds'] for ms in map_stats.values() if ms["name"] == map_played])

            multikills = total_triple_k * 9 + total_quadra_k * 16 + total_penta_k * 25
            multikills_score += (multikills /
                                 total_rounds) if total_rounds != 0 else AVERAGE_RMK
        mean_multikills_score.append(multikills_score/5)
    return mean_multikills_score


def get_mean_rating_on_map(data, team):
    mean_rating = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        rating = 0
        for p in team_data:
            map_stats = p["mapStats"]

            total_kills = sum(
                [ms['kills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_deaths = sum(
                [ms['deaths'] for ms in map_stats.values() if ms["name"] == map_played])
            total_assists = sum(
                [ms['assists'] for ms in map_stats.values() if ms["name"] == map_played])
            total_mvps = sum(
                [ms['mvps'] for ms in map_stats.values() if ms["name"] == map_played])
            total_triple_k = sum(
                [ms['tripleKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_quadra_k = sum(
                [ms['quadraKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_penta_k = sum(
                [ms['pentaKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_rounds = sum(
                [ms['rounds'] for ms in map_stats.values() if ms["name"] == map_played])

            if total_rounds == 0:
                # TODO: review better value... impute mean
                rating += 1
            else:
                rating += compute_rating(total_kills, total_deaths, total_triple_k,
                                         total_quadra_k, total_penta_k, total_assists,
                                         total_mvps, total_rounds)

        mean_rating.append(rating/5)
    return mean_rating


def get_mean_matches_map_preference(data, team):
    mean_preference = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        players_preference = 0
        for p in team_data:
            map_stats = p["mapStats"]
            total_matches = sum([ms['matches'] for ms in map_stats.values()])
            total_matches_on_map = sum(
                [ms['matches'] for ms in map_stats.values() if ms["name"] == map_played])
            players_preference += total_matches_on_map / \
                total_matches if total_matches else 0.125
        mean_preference.append(players_preference/5)
    return mean_preference


def get_mean_winrate_map_preference(data, team):
    mean_winrate = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        player_winrate_map_preference = 0
        for p in team_data:
            map_stats = p["mapStats"]

            total_matches = sum([ms['matches'] for ms in map_stats.values()])
            total_wins = sum([ms['wins'] for ms in map_stats.values()])
            total_matches_on_map = sum(
                [ms['matches'] for ms in map_stats.values() if ms["name"] == map_played])
            total_wins_on_map = sum(
                [ms['wins'] for ms in map_stats.values() if ms["name"] == map_played])

            player_winrate = (
                total_wins / total_matches) if total_matches != 0 else 0.5
            player_winrate_on_map = (
                total_wins_on_map / total_matches_on_map) if total_matches_on_map != 0 else 0.5
            player_winrate_map_preference += (
                player_winrate_on_map / player_winrate) if player_winrate != 0 else 1

        mean_winrate.append(player_winrate_map_preference/5)
    return mean_winrate


def get_mean_kd_map_preference(data, team):
    mean_kd = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        player_kd_preference = 0
        for p in team_data:
            map_stats = p["mapStats"]

            total_kills = sum([ms['kills'] for ms in map_stats.values()])
            total_deaths = sum([ms['deaths'] for ms in map_stats.values()])
            total_kills_on_map = sum(
                [ms['kills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_deaths_on_map = sum(
                [ms['deaths'] for ms in map_stats.values() if ms["name"] == map_played])

            player_kd = (
                total_kills / total_deaths) if total_deaths != 0 else 1
            player_kd_on_map = (
                total_kills_on_map / total_deaths_on_map) if total_deaths_on_map != 0 else 1
            player_kd_preference += (player_kd_on_map /
                                     player_kd) if player_kd != 0 else 1

        mean_kd.append(player_kd_preference/5)
    return mean_kd


def get_mean_multikills_score_map_preference(data, team):
    mean_multikills_score = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        multikills_score_preference = 0
        for p in team_data:
            map_stats = p["mapStats"]

            total_triple_k_on_map = sum(
                [ms['tripleKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_quadra_k_on_map = sum(
                [ms['quadraKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_penta_k_on_map = sum(
                [ms['pentaKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_rounds_on_map = sum(
                [ms['rounds'] for ms in map_stats.values() if ms["name"] == map_played])
            total_triple_k = sum([ms['tripleKills']
                                  for ms in map_stats.values()])
            total_quadra_k = sum([ms['quadraKills']
                                  for ms in map_stats.values()])
            total_penta_k = sum([ms['pentaKills']
                                 for ms in map_stats.values()])
            total_rounds = sum([ms['rounds'] for ms in map_stats.values()])

            total_multikills = total_triple_k * 9 + total_quadra_k * 16 + total_penta_k * 25
            total_multikills_score = (
                total_multikills / total_rounds) if total_rounds != 0 else AVERAGE_RMK

            total_multikills_on_map = total_triple_k_on_map * 9 + \
                total_quadra_k_on_map * 16 + total_penta_k_on_map * 25
            total_multikills_score_on_map = (
                total_multikills_on_map / total_rounds_on_map) if total_rounds_on_map != 0 else AVERAGE_RMK

            multikills_score_preference += (total_multikills_score_on_map /
                                            total_multikills_score) if total_multikills_score != 0 else 1
        mean_multikills_score.append(multikills_score_preference/5)
    return mean_multikills_score


def get_mean_rating_map_preference(data, team):
    mean_rating = []
    for team_data, map_played in zip(data[team], data["mapPlayed"]):
        rating_preference = 0
        for p in team_data:
            map_stats = p["mapStats"]

            total_kills_on_map = sum(
                [ms['kills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_deaths_on_map = sum(
                [ms['deaths'] for ms in map_stats.values() if ms["name"] == map_played])
            total_assists_on_map = sum(
                [ms['assists'] for ms in map_stats.values() if ms["name"] == map_played])
            total_mvps_on_map = sum(
                [ms['mvps'] for ms in map_stats.values() if ms["name"] == map_played])
            total_triple_k_on_map = sum(
                [ms['tripleKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_quadra_k_on_map = sum(
                [ms['quadraKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_penta_k_on_map = sum(
                [ms['pentaKills'] for ms in map_stats.values() if ms["name"] == map_played])
            total_rounds_on_map = sum(
                [ms['rounds'] for ms in map_stats.values() if ms["name"] == map_played])

            total_kills = sum([ms['kills'] for ms in map_stats.values()])
            total_deaths = sum([ms['deaths'] for ms in map_stats.values()])
            total_assists = sum([ms['assists'] for ms in map_stats.values()])
            total_mvps = sum([ms['mvps'] for ms in map_stats.values()])
            total_triple_k = sum([ms['tripleKills']
                                  for ms in map_stats.values()])
            total_quadra_k = sum([ms['quadraKills']
                                  for ms in map_stats.values()])
            total_penta_k = sum([ms['pentaKills']
                                 for ms in map_stats.values()])
            total_rounds = sum([ms['rounds'] for ms in map_stats.values()])

            if total_rounds == 0:
                # TODO: review better value... impute mean
                total_rating += 1
            else:
                total_rating = compute_rating(total_kills, total_deaths, total_triple_k,
                                              total_quadra_k, total_penta_k, total_assists,
                                              total_mvps, total_rounds)
            if total_rounds_on_map == 0:
                # TODO: review better value... impute mean
                total_rating_on_map += 1
            else:
                total_rating_on_map = compute_rating(total_kills_on_map, total_deaths_on_map, total_triple_k_on_map,
                                                     total_quadra_k_on_map, total_penta_k_on_map, total_assists_on_map,
                                                     total_mvps_on_map, total_rounds_on_map)
            rating_preference += (total_rating_on_map /
                                  total_rating) if total_rating != 0 else 1
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
