from tqdm import tqdm
from src.data.constants import BATCH_SIZE, NUM_PREV_MATCHES
from src.utils.dirs import EXTERNAL_DATA_DIR, EXTERNAL_DATA_DIR_S, INTERIM_DATA_DIR, INTERIM_DATA_DIR_S
from src.db.connections import get_local_db
import pandas as pd

from src.utils.data_handlers import load_processable_match_ids


db = get_local_db()

players_coll = db['player']
matches_coll = db['match']
lifetime_stats_coll = db['player_lifetime_stats']


def set_previous_data(player, match_id, player_data):
    player_match_history = player_data["matchHistory"]
    match_history_ids = [m['id'] for m in player_match_history]
    match_index = match_history_ids.index(match_id)
    player["previousMatches"] = match_history_ids[match_index +
                                                  1:match_index+NUM_PREV_MATCHES+1]


def set_player_data(player, players_data):
    player_id = player["id"]
    players_data_fields = ['activatedAt', 'steamCreatedAt',
                           'updatedAt', 'csgoId', 'verified']

    # check if player already in data, if not retrieve from DB and store
    player_data = players_data.get(player_id, None)
    if not player_data:
        player_data = players_coll.find_one({"_id": player_id},
                                            {*players_data_fields, "matchHistory"})
        players_data[player_data["_id"]] = player_data

    for player_field in players_data_fields:
        player[player_field] = players_data[player_id][player_field]

    return player_data


def build_interim_dataset(interim_data_dir, external_data_dir, is_complete):
    matches_processed = []
    players_data = {}

    match_ids = load_processable_match_ids(external_data_dir)

    matches_to_process = matches_coll.find(
        {"_id": {"$in": match_ids}}, {"teams": 0})

    batch_number = 0

    for index, match in enumerate(tqdm(matches_to_process, total=len(match_ids))):
        # Get all ids of the players in the match
        players_ids = {p['id']
                       for team in (match['teamA'], match['teamB']) for p in team}

        lifetime_stats = lifetime_stats_coll.find({
            "matchId": match['_id'],
            "playerId": {"$in": list(players_ids)}})

        lifetime_data = {lt["playerId"]: lt for lt in lifetime_stats}

        for team in ["teamA", "teamB"]:
            for player in match[team]:
                player_id = player["id"]

                player_data = set_player_data(player, players_data)
                player["mapStats"] = lifetime_data[player_id]["mapStats"]

                if is_complete:
                    set_previous_data(player, match['_id'], player_data)

        matches_processed.append(match)

        if index % BATCH_SIZE == 0 and index > 0:
            pd.DataFrame(matches_processed).to_csv(
                f'{str(interim_data_dir)}/batch_{batch_number}.csv')
            batch_number += 1
            matches_processed.clear()

    if matches_processed:
        pd.DataFrame(matches_processed).to_csv(
            f'{str(interim_data_dir)}/batch_{batch_number}.csv')


if __name__ == '__main__':
    # Build Interim Dataset Complete
    build_interim_dataset(INTERIM_DATA_DIR,
                          EXTERNAL_DATA_DIR,
                          is_complete=True)

    # Build Interim Dataset Simplified
    build_interim_dataset(INTERIM_DATA_DIR_S,
                          EXTERNAL_DATA_DIR_S,
                          is_complete=False)
