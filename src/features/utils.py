def add_feature(data, function, **kwargs):
    teams = ("A", "B")

    # returns (everything beyond get_)
    feature = '_'.join(function.__name__.split("_")[1:])

    for team in teams:
        feature_team = '_'.join([feature, team])
        data[feature_team] = function(data, "team" + team, **kwargs)

    data["dif_" + feature] = data[feature + "_A"] - data[feature + "_B"]


def get_team_rounds(score_string):
    return [int(r) for r in score_string.split("/")]


def get_player_won_the_match(match, player_id):
    is_on_team_A = True if player_id in [player['id']
                                         for player in match['teamA']] else False
    team_rounds = get_team_rounds(match["score"])

    if (is_on_team_A and team_rounds[0] > team_rounds[1]) or \
            (not is_on_team_A and team_rounds[1] > team_rounds[0]):
        return 1
    else:
        return 0
