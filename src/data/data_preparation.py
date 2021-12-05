from src.utils.dirs import (INTERIM_DATA_DIR_S, COMPLEMENTARY_DATA_DIR_S,
                            RAW_DATA_DIR_S, INTERIM_DATA_DIR, COMPLEMENTARY_DATA_DIR, RAW_DATA_DIR)
from src.data.processable_matches import processable_matches_simple, processable_matches_complete
from src.data.build_raw import build_raw_dataset
from src.data.build_interim import build_interim_dataset
from src.data.create_lifetime_stats import create_all_lifetime_stats
from src.data.performance_indicators import create_performance_indicators


def data_preparation():
    create_all_lifetime_stats()
    processable_matches_complete()
    processable_matches_simple()

    create_performance_indicators()

    build_raw_dataset(RAW_DATA_DIR, COMPLEMENTARY_DATA_DIR)
    build_interim_dataset(
        INTERIM_DATA_DIR, COMPLEMENTARY_DATA_DIR, is_complete=True)

    build_raw_dataset(RAW_DATA_DIR_S, COMPLEMENTARY_DATA_DIR_S)
    build_interim_dataset(INTERIM_DATA_DIR_S,
                          COMPLEMENTARY_DATA_DIR_S, is_complete=False)


if __name__ == '__main__':
    data_preparation()
