import mlflow

from sklearn.model_selection import train_test_split
from src.models.hp_tuning import get_best_params
from src.models.available_models import MODELS
from src.models.scalers import get_scaled_data

from mlflow.models.signature import infer_signature

from src.utils.utils import SEED, filter_by_key
from src.models.experiments import set_experiment
from mlflow.entities.view_type import ViewType


class ProductionTrainer:
    def __init__(self, run_name, clf, params, is_complete):
        self.run_name = self.build_run_name(run_name)
        self.clf = clf
        self.params = params
        self.is_complete = is_complete
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        set_experiment(is_complete)

    @staticmethod
    def build_run_name(run_name):
        return f'best_{run_name}'

    def set_data(self, data):
        self.X = data["X"]
        self.y = data["y"]
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_test = data["X_test"]
        self.y_test = data["y_test"]

    def train(self):
        with mlflow.start_run(run_name=self.run_name):
            clf = self.clf(**self.params, random_state=SEED)
            clf.fit(self.X_train, self.y_train)

            signature = infer_signature(self.X_train, self.y_train)
            mlflow.log_params(self.params)
            mlflow.sklearn.log_model(clf, "model", signature=signature)

            mlflow.sklearn.eval_and_log_metrics(
                clf, self.X_train, self.y_train, prefix="train_")
            mlflow.sklearn.eval_and_log_metrics(
                clf, self.X_test, self.y_test, prefix="test_")
            mlflow.set_tag("num_train_observations", len(self.X_train))
            mlflow.set_tag("num_total_observations", len(self.X))

            # Train Final model with full data and log exclusively the model
            clf.fit(self.X, self.y)
            mlflow.sklearn.log_model(clf, "final_model", signature=signature)


def load_best_model(name):
    best_run_name = ProductionTrainer.build_run_name(name)

    parent_run = mlflow.search_runs(
        filter_string=f'tags.mlflow.runName="{best_run_name}"',
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    if parent_run.empty:
        return
    run_id = parent_run.iloc[0].run_id

    logged_model = f'runs:/{run_id}/final_model'

    return mlflow.sklearn.load_model(logged_model)


def create_best_models(selected_models, is_complete):
    set_experiment(is_complete)

    models = filter_by_key(selected_models, MODELS)
    X_scaled, y = get_scaled_data(is_complete=is_complete)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=SEED)

    data = {
        "X": X_scaled,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    for name, clf in models.items():
        best_params = get_best_params(name)
        trainer = ProductionTrainer(name, clf, best_params, data)
        trainer.set_data(data)
        trainer.train()


if __name__ == '__main__':
    '''
    Implement code to train a specific algorithm with custom params and make it ready to be used in production.
    Store the model as an artifact in mlflow, along with its params and metrics.
    '''
    create_best_models(selected_models='all', is_complete=True)
    # create_best_models(selected_models='all', is_complete=False)
