from collections import defaultdict
from src.data.constants import NUM_PREV_MATCHES
from src.db.connections import get_local_db
from tqdm import tqdm
from src.utils.dirs import EXTERNAL_DATA_DIR, EXTERNAL_DATA_DIR_S

from src.utils.data_handlers import store_processable_match_ids

db = get_local_db()

players_coll = db['player']
matches_coll = db['match']
lifetime_stats_coll = db['player_lifetime_stats']


def set_match_history_ready(matches_ready):
    all_players = players_coll.find({})

    for p in tqdm(all_players, total=players_coll.estimated_document_count()):
        prev_matches = sorted(p["matchHistory"], key=lambda x: x["startTime"])
        prev_matches_ids = [m["id"] for m in prev_matches]

        matches_ids_in_db = set(matches_coll.distinct(
            "_id", {"_id": {"$in": prev_matches_ids}}))
        missing_decay = 0
        for index, m in enumerate(prev_matches_ids):
            match_id = m["id"]
            if match_id not in matches_ids_in_db:
                missing_decay = NUM_PREV_MATCHES
                continue

            if missing_decay > 0:
                missing_decay -= 1
            elif missing_decay == 0 and index > NUM_PREV_MATCHES - 1:
                matches_ready[match_id]["match_history"] += 1


def set_lifetime_ready(matches_ready):
    all_lifetime_stats = lifetime_stats_coll.find({})

    for l in tqdm(all_lifetime_stats, total=lifetime_stats_coll.estimated_document_count()):
        matches_ready[l["matchId"]]["lifetime_stats"] += 1


def processable_matches_complete():
    matches_ready = defaultdict(
        lambda: {"match_history": 0, "lifetime_stats": 0})

    set_match_history_ready(matches_ready)
    set_lifetime_ready(matches_ready)

    # Filter match ids to include only those who have full data for all ten players
    match_ids = [m_id for m_id, rd in matches_ready.items(
    ) if rd["match_history"] == 10 and rd["lifetime_stats"] == 10]

    store_processable_match_ids(match_ids, EXTERNAL_DATA_DIR)


def processable_matches_simple():
    matches_ready = defaultdict(lambda: {"lifetime_stats": 0})

    set_lifetime_ready(matches_ready)

    # Filter match ids to include only those who have lifetime data for all ten player
    match_ids = [m_id for m_id, rd in matches_ready.items()
                 if rd["lifetime_stats"] == 10]

    store_processable_match_ids(match_ids, EXTERNAL_DATA_DIR_S)


def main():
    '''
    Even though there are more than one million matches stored in DB,
    not all are qualified to be part of the dataset.

    A valid processable match should have complete player data for every participant.
    This includes the lifetime stats before the beginning of the match as well as his *10* previous matches.

    - `match_history`: the match id is in the player's match history and the previous 10 matches are in DB
    - `lifetime_stats`: the lifetime stats of the player regarding the match are available

    The `match_history` and `lifetime_stats` are initially set to 0
    and incremented each time the conditions are met for one player.
    '''
    processable_matches_complete()
    processable_matches_simple()


if __name__ == '__main__':
    main()
