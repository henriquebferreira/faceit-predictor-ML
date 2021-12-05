'''
TODO:Rewrite code as an airflow DAG
Contains all the code from end to end to create the ML models.
'''


from src.models.experiments import init_experiments
from src.models.productionize import create_best_models
from src.models.scalers import build_scaler
from src.models.hp_tuning import hyperparameter_tuning
from src.features.featurize import featurize
from src.data.data_preparation import data_preparation
from sklearn.preprocessing import StandardScaler


def main_workflow():
    data_preparation()

    featurize(is_complete=True)
    featurize(is_complete=False)

    init_experiments()

    for is_complete in (True, False):
        build_scaler("standard_scaler",
                     StandardScaler,
                     is_complete=is_complete)
        hyperparameter_tuning(selected_models='all', is_complete=is_complete)
        create_best_models(selected_models='all', is_complete=is_complete)


if __name__ == '__main__':
    main_workflow()
