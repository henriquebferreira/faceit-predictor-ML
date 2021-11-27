from src.features.match_features import add_match_features
from src.features.lifetime_features import add_lifetime_features
from src.features.date_features import add_date_features
from src.features.previous_matches_features import add_previous_matches_features
from src.utils.data_handlers import read_data_iter
from src.utils.dirs import PROCESSED_DATA_DIR, PROCESSED_DATA_DIR_S


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


def featurize(is_complete):
    processed_folder = PROCESSED_DATA_DIR if is_complete else PROCESSED_DATA_DIR_S
    add_features = add_all_features if is_complete else add_simplified_features

    batch_data_gen = read_data_iter("interim", is_complete)

    for index, data in enumerate(batch_data_gen):
        data = add_features(data)
        data = select_features(data)
        data.to_feather(f'{str(processed_folder)}/batch_{index}.feather')


if __name__ == '__main__':
    featurize(is_complete=True)
    featurize(is_complete=False)
