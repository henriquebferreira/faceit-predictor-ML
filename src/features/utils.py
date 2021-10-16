def get_team_rounds(score_string):
    rounds = score_string.split("/")
    team_rounds = list(map(lambda r: int(r), rounds))
    return team_rounds[0], team_rounds[1]
