from tqdm import tqdm
from src.data.constants import BATCH_SIZE
from src.utils.dirs import COMPLEMENTARY_DATA_DIR, COMPLEMENTARY_DATA_DIR_S, RAW_DATA_DIR, RAW_DATA_DIR_S
from src.db.connections import get_local_ingestor_db
import pandas as pd

from src.utils.data_handlers import load_processable_match_ids


db = get_local_ingestor_db()

players_coll = db['player']
matches_coll = db['match']
lifetime_stats_coll = db['player_lifetime_stats']


def build_raw_dataset(raw_data_dir, external_data_dir):
    match_ids = load_processable_match_ids(external_data_dir)
    num_matches = len(match_ids)

    print("Number of processable matches", num_matches)

    num_batches = (num_matches // BATCH_SIZE) + 1

    for batch_index in tqdm(range(num_batches), desc=f"Processing batches of {BATCH_SIZE}"):
        matches_to_process = list(matches_coll
                                  .find({"_id": {"$in": match_ids}}, {"teams": 0})
                                  .skip(batch_index*BATCH_SIZE)
                                  .limit(BATCH_SIZE))

        pd.DataFrame(matches_to_process).to_csv(
            f'{str(raw_data_dir)}/batch_{batch_index}.csv')


if __name__ == '__main__':
    # Build Raw Dataset Complete
    build_raw_dataset(RAW_DATA_DIR, COMPLEMENTARY_DATA_DIR)

    # Build Raw Dataset Simplified
    build_raw_dataset(RAW_DATA_DIR_S, COMPLEMENTARY_DATA_DIR_S)
