from statistics import mean



def get_mean_matches(match, team, **kwargs):
    team_matches = []

    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    for player_id in team_players_ids:
        player = players_df[players_df['_id'] == player_id].iloc[0]
        player_matches = []
        for map_name, map_stats in player['mapStats'].items():
            player_matches.append(map_stats['matches'])
        team_matches.append(sum(player_matches))

    return mean(team_matches)

def get_mean_winrate(match, team, **kwargs):
    team_matches = []

    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    for player_id in team_players_ids:
        player = players_df[players_df['_id'] == player_id].iloc[0]
        player_matches = []
        for map_name, map_stats in player['mapStats'].items():
            player_matches.append(map_stats['matches'])
        team_matches.append(sum(player_matches))

    return mean(team_matches)

def get_mean_kd(match, team, **kwargs):
    team_matches = []

    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    for player_id in team_players_ids:
        player = players_df[players_df['_id'] == player_id].iloc[0]
        player_matches = []
        for map_name, map_stats in player['mapStats'].items():
            player_matches.append(map_stats['matches'])
        team_matches.append(sum(player_matches))

    return mean(team_matches)

def get_mean_matches_on_map(match, team, **kwargs):
    team_matches = []

    map_played = match['mapPlayed']
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    for player_id in team_players_ids:
        player = players_df[players_df['_id'] == player_id].iloc[0]
        if map_played in player['mapStats']:
            team_matches.append(player['mapStats'][map_played]['matches'])
        else:
            team_matches.append(0)

    return mean(team_matches)


def get_mean_winrate_on_map(match, team, **kwargs):
    win_rates = []

    map_played = match['mapPlayed']
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    for player_id in team_players_ids:
        player = players_df[players_df['_id'] == player_id].iloc[0]
        if map_played in player['mapStats']:
            if player['mapStats'][map_played]['matches'] == 0:
                continue
            player_win_rate = player['mapStats'][map_played]['wins'] / player['mapStats'][map_played]['matches']
            win_rates.append(player_win_rate)
        else:
            # Instead of imputing 0, compute the player's average win rate for the other maps
            win_rates.append(0)

    return mean(win_rates)


def get_mean_kd_on_map(match, team, **kwargs):
    kds = []

    map_played = match['mapPlayed']
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    for player_id in team_players_ids:
        player = players_df[players_df['_id'] == player_id].iloc[0]
        if map_played in player['mapStats']:
            player_kd = player['mapStats'][map_played]['kills'] / player['mapStats'][map_played]['deaths']
            kds.append(player_kd)
        else:
            # Instead of imputing 0, compute the player's average win rate for the other maps
            kds.append(0)

    return mean(kds)


def get_mean_matches_map_preference(match, team, **kwargs):
    team_matches_ratio = []

    map_played = match['mapPlayed']
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    for player_id in team_players_ids:
        player = players_df[players_df['_id'] == player_id].iloc[0]
        player_matches = []
        player_matches_on_map = 0
        for map_name, map_stats in player['mapStats'].items():
            if map_name == map_played:
                player_matches_on_map = map_stats['matches']
            player_matches.append(map_stats['matches'])
        if not player_matches:
            team_matches_ratio.append(0)
        elif mean(player_matches) == 0:
            team_matches_ratio.append(0)
        else:
            team_matches_ratio.append(player_matches_on_map / mean(player_matches))

    return mean(team_matches_ratio)


def get_mean_winrate_map_preference(match, team, **kwargs):
    team_winrates_ratio = []

    map_played = match['mapPlayed']
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    for player_id in team_players_ids:
        player = players_df[players_df['_id'] == player_id].iloc[0]
        player_winrates = []
        player_winrate_on_map = 0
        for map_name, map_stats in player['mapStats'].items():
            if map_stats['matches'] == 0:
                continue
            if map_name == map_played:
                player_winrate_on_map = map_stats['wins'] / map_stats['matches']
            player_winrates.append(map_stats['wins'] / map_stats['matches'])
        if not player_winrates:
            team_winrates_ratio.append(0)
        else:
            if mean(player_winrates) == 0:
                team_winrates_ratio.append(0)
            else:
                team_winrates_ratio.append(player_winrate_on_map / mean(player_winrates))

    return mean(team_winrates_ratio)


def get_mean_kd_map_preference(match, team, **kwargs):
    team_kds_ratio = []

    map_played = match['mapPlayed']
    team_players_ids = [p['id'] for p in match[team]]
    players_df = kwargs['players']

    for player_id in team_players_ids:
        player = players_df[players_df['_id'] == player_id].iloc[0]
        player_kds = []
        player_kd_on_map = 0
        for map_name, map_stats in player['mapStats'].items():
            if map_name == map_played:
                player_kd_on_map = map_stats['kills'] / map_stats['deaths']
            if map_stats['deaths'] == 0:
                player_kds.append(0)
            else:
                player_kds.append(map_stats['kills'] / map_stats['deaths'])
        if not player_kds:
            team_kds_ratio.append(0)
        else:
            team_kds_ratio.append(player_kd_on_map / mean(player_kds))

    return mean(team_kds_ratio)
