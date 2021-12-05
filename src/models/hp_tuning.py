
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
from statistics import mean
import mlflow
from sklearn.model_selection import cross_validate, train_test_split
from src.models.experiments import set_experiment
from src.models.search_spaces import SEARCH_SPACES
from src.models.scalers import get_scaled_data
from src.models.available_models import MODELS
from fastnumbers import fast_real
from src.utils.utils import SEED, filter_by_key
from mlflow.entities.view_type import ViewType
from hyperopt.pyll.stochastic import sample


class HyperparametersTuner:
    SCORING_METRICS = ('accuracy', 'roc_auc')
    CV_FOLDS = 5

    def __init__(self, run_name, clf, search_space, is_complete, max_evals=None, sample_size=None):
        self.run_name = self.build_run_name(run_name)
        self.clf = clf
        self.search_space = search_space
        self.max_evals = max_evals
        self.sample_size = sample_size
        self.X_train = None
        self.y_train = None

        set_experiment(is_complete)

    @staticmethod
    def build_run_name(run_name):
        return f'tuned_{run_name}'

    def set_train_data(self, X_train, y_train):
        self.X_train = X_train.iloc[:self.sample_size]
        self.y_train = y_train.iloc[:self.sample_size]

    def _validate_params(self, params):
        try:
            clf = self.clf(**params)
            clf.fit(self.X_train.iloc[:100], self.y_train.iloc[:100])
            return True
        except:
            return False

    def _train_model(self, params):
        while not self._validate_params(params):
            params = sample(self.search_space)

        with mlflow.start_run(nested=True, run_name=self.clf.__name__):
            clf = self.clf(**params, random_state=SEED)

            scores = cross_validate(clf,
                                    self.X_train,
                                    self.y_train,
                                    scoring=HyperparametersTuner.SCORING_METRICS,
                                    cv=HyperparametersTuner.CV_FOLDS,
                                    return_train_score=True)
            cv_mean_metrics = {f'mean_{k}': mean(v) for k, v in scores.items()}
            mlflow.log_metrics(cv_mean_metrics)
            mlflow.log_params(params)

            # Set the loss to -1*auc_score so fmin maximizes the auc_score
            return {'status': STATUS_OK, 'loss': -1*cv_mean_metrics["mean_test_roc_auc"], 'params': params}

    def optimize(self):
        trials = Trials()

        # Run fmin within an MLflow run context so that each hyperparameter
        # configuration is logged as a child run of a parent
        with mlflow.start_run(run_name=self.run_name):
            fmin(
                fn=self._train_model,
                space=self.search_space,
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=trials,
            )
            best_trial = trials.best_trial
            best_params = best_trial["result"]["params"]
            mlflow.log_params(best_params)
            mlflow.log_metric("best_test_roc_auc",
                              -best_trial["result"]["loss"])
            mlflow.set_tag("model", self.clf.__name__)
            mlflow.set_tag('seed', SEED)
            mlflow.set_tag("sample_size", len(self.X_train))


def get_best_params(name):
    tuned_run_name = HyperparametersTuner.build_run_name(name)

    parent_run = mlflow.search_runs(
        filter_string=f'tags.mlflow.runName="{tuned_run_name}"',
        order_by=["metrics.best_test_roc_auc DESC"],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    if parent_run.empty:
        return

    parent_run = parent_run.iloc[0]

    params_cols = [c for c in parent_run.index if c.startswith("params.")]
    params = parent_run[params_cols].to_dict()
    formatted_params = {}
    for k, v in params.items():
        key_name = k.split("params.")[-1]
        converted_value = fast_real(v)
        if converted_value == 'True':
            converted_value = True
        elif converted_value == 'False':
            converted_value = False
        formatted_params[key_name] = converted_value
    return formatted_params


def hyperparameter_tuning(selected_models, is_complete):
    models = filter_by_key(selected_models, MODELS)
    search_spaces = filter_by_key(selected_models, SEARCH_SPACES)

    X_scaled, y = get_scaled_data(is_complete=is_complete)

    # Split data
    X_train, _, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.3, random_state=SEED)

    for model_key in models.keys():
        optimizer = HyperparametersTuner(model_key,
                                         models[model_key],
                                         search_spaces[model_key],
                                         is_complete=is_complete,
                                         sample_size=10000,
                                         max_evals=30)
        optimizer.set_train_data(X_train, y_train)
        optimizer.optimize()


if __name__ == '__main__':
    hyperparameter_tuning(selected_models="all", is_complete=True)
    # hyperparameter_tuning(selected_models="all", is_complete=False)
