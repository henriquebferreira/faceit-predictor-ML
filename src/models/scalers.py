from src.models.experiments import set_experiment
from sklearn.preprocessing import StandardScaler
from src.features.featurize import select_features
from src.utils.data_handlers import read_data
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd


def build_scaler(name, model_class, is_complete):
    set_experiment(is_complete)
    mlflow.sklearn.autolog()

    data = read_data("processed", is_complete=is_complete)
    data = select_features(data)
    X = data.drop(columns=["winner", "_id"])

    with mlflow.start_run(run_name=name):
        ss = model_class()
        ss.fit(X)
        signature = infer_signature(X, X)
        mlflow.sklearn.log_model(ss, name, signature=signature)


def load_scaler(name, is_complete):
    set_experiment(is_complete)
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName='{name}'")
    run_id = runs.iloc[0].run_id

    return mlflow.sklearn.load_model(f'runs:/{run_id}/{name}')


def get_scaled_data(is_complete):
    # Load Standard Scaler
    sc = load_scaler("standard_scaler", is_complete=is_complete)

    # Load Pre-processed data
    data = read_data("processed", is_complete=is_complete)
    data = select_features(data)
    X = data.drop(columns=["winner", "_id"])
    y = data["winner"]

    # Scale data and re-create dataframe with same columns
    X_scaled = pd.DataFrame(sc.transform(X), columns=X.columns)

    return X_scaled, y


def main():
    build_scaler("standard_scaler", StandardScaler, is_complete=True)
    build_scaler("standard_scaler", StandardScaler, is_complete=False)


if __name__ == '__main__':
    main()
