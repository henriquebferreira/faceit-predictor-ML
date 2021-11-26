from src.features.match_features import add_match_features
from src.features.lifetime_features import add_lifetime_features
from src.features.date_features import add_date_features
from src.features.previous_matches_features import add_previous_matches_features
from utils.data_handlers import read_data_iter
from utils.dirs import PROCESSED_DATA_DIR, PROCESSED_DATA_DIR_S


def add_all_features(data_matches):
    add_match_features(data_matches)
    add_lifetime_features(data_matches)
    add_date_features(data_matches)
    add_previous_matches_features(data_matches)

    return data_matches


def add_simplified_features(data_matches):
    add_match_features(data_matches)
    add_lifetime_features(data_matches)
    add_date_features(data_matches)

    return data_matches


def select_features(data):
    selected_columns = ["_id", "winner", "match_mean_elo",
                        "5v5_free_queue", "5v5_premium_queue"]
    dif_columns = [c for c in data.columns if c.startswith("dif_")]
    team_feature_columns = [
        c for c in data.columns if c.endswith("_A") or c.endswith("_B")]
    map_dummies_columns = [
        c for c in data.columns if c.startswith("map_dummies")]
    entity_dummies_columns = [
        c for c in data.columns if c.startswith("entity_dummies")]

    selected_columns.extend(dif_columns)
    selected_columns.extend(team_feature_columns)
    selected_columns.extend(map_dummies_columns)
    selected_columns.extend(entity_dummies_columns)
    return data[selected_columns]


def featurize_complete():
    batch_data_gen = read_data_iter("interim", is_complete=True)
    for index, data in enumerate(batch_data_gen):
        data_featurized = add_all_features(data)
        data_featurized = select_features(data_featurized)
        data_featurized.to_feather(
            f'{str(PROCESSED_DATA_DIR)}/batch_{index}.feather')


def featurize_simplified():
    batch_data_gen = read_data_iter("interim", is_complete=False)
    for index, data in enumerate(batch_data_gen):
        data_featurized = add_simplified_features(data)
        data_featurized = select_features(data_featurized)
        data_featurized.to_feather(
            f'{str(PROCESSED_DATA_DIR_S)}/batch_{index}.feather')


if __name__ == '__main__':
    featurize_complete()
    featurize_simplified()
