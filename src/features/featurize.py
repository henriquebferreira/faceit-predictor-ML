from src.features.match_features import add_match_features
from src.features.lifetime_features import add_lifetime_features
from src.features.date_features import add_date_features
from src.features.previous_matches_features import add_previous_matches_features


def add_all_features(data_matches):
    add_match_features(data_matches)
    add_lifetime_features(data_matches)
    add_date_features(data_matches)

    data_matches = data_matches.apply(add_previous_matches_features, axis=1)
    return data_matches


def select_features(data):
    selected_columns = ["_id", "winner", "match_mean_elo",
                        "5v5_free_queue", "5v5_premium_queue"]
    dif_columns = [c for c in data.columns if c.startswith("dif_")]
    map_dummies_columns = [
        c for c in data.columns if c.startswith("map_dummies")]
    entity_dummies_columns = [
        c for c in data.columns if c.startswith("entity_dummies")]

    selected_columns.extend(dif_columns)
    selected_columns.extend(map_dummies_columns)
    selected_columns.extend(entity_dummies_columns)
    return data[selected_columns]


def create_features(data_matches):
    featurized_matches = add_all_features(data_matches)
    featurized_matches = select_features(featurized_matches)

    return featurized_matches
