import json

# Load performance statistics that were previously computed
with open('D:\\Portfolio\\faceit-predictor\\ML\\data\\external\\performance_statistics.json') as fp:
    performance_statistics = json.load(fp)

AVERAGE_KPR = performance_statistics["meanKPR"]
AVERAGE_SPR = performance_statistics["meanSPR"]
AVERAGE_RMK = performance_statistics["meanMKPR"]
AVERAGE_APR = performance_statistics["meanAPR"]
AVERAGE_MVPPR = performance_statistics["meanMVPPR"]


def add_feature(data, function, **kwargs):
    teams = ("A", "B")

    # returns (everything beyond get_)
    feature = '_'.join(function.__name__.split("_")[1:])

    for team in teams:
        feature_team = '_'.join([feature, team])
        data[feature_team] = function(data, "team" + team, **kwargs)

    data["dif_" + feature] = data[feature + "_A"] - data[feature + "_B"]


def compute_feature(data, features_data, function, **kwargs):
    teams = ("A", "B")

    # returns (everything beyond get_)
    feature = '_'.join(function.__name__.split("_")[1:])
    for team in teams:
        feature_team = '_'.join([feature, team])
        feature_value = function(data, "team" + team, **kwargs)
        features_data[feature_team].append(feature_value)


def get_team_rounds(score_string):
    return [int(r) for r in score_string.split("/")]


def get_player_won_the_match(match, player_id):
    players_team_A = [player['id'] for player in match['teamA']]
    is_on_team_A = True if player_id in players_team_A else False

    team_rounds = get_team_rounds(match["score"])
    if (is_on_team_A and team_rounds[0] > team_rounds[1]) or \
            (not is_on_team_A and team_rounds[1] > team_rounds[0]):
        return 1
    else:
        return 0


def summation_by(data, field):
    summation = 0
    for d in data:
        summation += d.get(field, 0)
    return summation


def compute_rating(kills, deaths, triple_k, quadra_k, penta_k, assists, mvps, rounds):
    if rounds == 0:
        return 1

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
