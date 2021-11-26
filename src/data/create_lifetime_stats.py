from pymongo.errors import PyMongoError
from tqdm import tqdm
from pymongo import DESCENDING
from src.db.connections import get_local_db

db = get_local_db()

players_coll = db['player']
matches_coll = db['match']
lifetime_stats_coll = db['player_lifetime_stats']


def subtract_scoreboard_stats(player_lifetime, player_scoreboard):
    for k in player_scoreboard.keys():
        player_lifetime[k] -= player_scoreboard.get(k, 0)


def get_won_the_match(player_id, match, team_rounds):
    player_ids_team_A = [p['id'] for p in match['teamA']]
    is_on_team_A = player_id in player_ids_team_A

    if (is_on_team_A and team_rounds[0] > team_rounds[1]) or \
            (not is_on_team_A and team_rounds[1] > team_rounds[0]):
        return 1
    else:
        return 0


def update_stats(next_lifetime_stats, player, match):
    player_id = player["_id"]
    score = match["score"]
    map_played = match['mapPlayed']

    # create a deep copy of previous map stats
    new_lifetime_stats = {k: v for k, v in next_lifetime_stats.items()}
    new_lifetime_stats[map_played]["matches"] -= 1

    team_rounds = [int(r) for r in score.split("/")]
    new_lifetime_stats[map_played]['rounds'] -= sum(team_rounds)

    won_the_match = get_won_the_match(player_id, match, team_rounds)
    new_lifetime_stats[map_played]['wins'] -= won_the_match

    players_of_match = [player for team in match['teams']
                        for player in team]
    player_stats_on_match = [
        p for p in players_of_match if p["id"] == player_id][0]["playerStats"]

    if not player_stats_on_match:
        return None

    subtract_scoreboard_stats(
        new_lifetime_stats[map_played], player_stats_on_match)

    return new_lifetime_stats


def get_next_lifetime_stats(player, match_id):
    player_id = player["_id"]

    # Match history is sorted in temporal descending order
    # The following matches are stored in the preceding indexes
    next_match_index = -1
    for index, m_history in enumerate(player["matchHistory"]):
        if m_history["id"] == match_id:
            next_match_index = index - 1
            break

    # If the match was the last to be played (1st one in match history),
    # then retrieve the current lifetime stats of the player
    if next_match_index < 0:
        next_lifetime_stats = player["mapStats"]
    else:
        previous_match = player["matchHistory"][next_match_index]
        next_lifetime_stats = lifetime_stats_coll.find_one(
            {"playerId": player_id,
             "matchId": previous_match["id"]},
            {"_id": 0, "mapStats": 1})

        if not next_lifetime_stats:
            raise PyMongoError("No previous lifetime stats")

        next_lifetime_stats = next_lifetime_stats["mapStats"]

    return next_lifetime_stats


def compute_new_lifetime_stats(player, match):
    player_id = player["_id"]
    match_id = match["_id"]
    match_start_time = match["startTime"]

    # return if match was played after player processing time
    matches_of_player = [m["id"] for m in player["matchHistory"]]
    if match_start_time > player["updatedAtIngestor"] or match_id not in matches_of_player:
        return None

    try:
        next_lifetime_stats = get_next_lifetime_stats(player, match_id)
    except:
        return None

    if match["mapPlayed"] not in next_lifetime_stats:
        return None

    new_stats = {}
    new_stats["matchId"] = match_id
    new_stats["playerId"] = player_id
    new_stats["startTime"] = match_start_time
    map_stats = update_stats(next_lifetime_stats, player, match)
    if not map_stats:
        return None
    new_stats["mapStats"] = map_stats

    return new_stats


def create_all_lifetime_stats():
    matches_cursor = matches_coll.find({}).sort("startTime", DESCENDING)

    for m in tqdm(matches_cursor, total=matches_coll.estimated_document_count()):
        # Get all ids of the players in the match
        players_ids = {player['id']
                       for team in m['teams'] for player in team}

        # Get the ids of the players whose lifetime stats
        # were already processed for this match
        players_ids_processed = set(lifetime_stats_coll.distinct("playerId", {
            "matchId": m['_id'],
            "playerId": {"$in": list(players_ids)}}))

        players_ids_to_process = players_ids - players_ids_processed
        if not players_ids_to_process:
            return

        players_to_process = players_coll.find(
            {"_id": {"$in": list(players_ids_to_process)}})

        lifetime_stats = [compute_new_lifetime_stats(
            player, m) for player in players_to_process]

        # Filter null values
        lifetime_stats = [x for x in lifetime_stats if x]
        if lifetime_stats:
            lifetime_stats_coll.insert_many(lifetime_stats)


def main():
    '''
    Initially only the most recent lifetime stats are stored in DB (`player.mapStats`).
    In order to have consistent player lifetime stats for each match and avoid repeating the process over again,
    the lifetime stats are processed once and stored in DB.

    To do so one must work backwards and continuously subtract the player stats on each match to the lifetime stats.
    '''

    create_all_lifetime_stats()


if __name__ == '__main__':
    main()
