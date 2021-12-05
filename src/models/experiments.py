import mlflow
import shutil
import os

COMPLETE_EXPERIMENT = "Complete"
SIMPLIFIED_EXPERIMENT = "Simplified"


def set_experiment(is_complete):
    mlflow.set_experiment(
        COMPLETE_EXPERIMENT if is_complete else SIMPLIFIED_EXPERIMENT)


def clear_experiments():
    for exp in mlflow.list_experiments():
        mlflow.delete_experiment(exp.experiment_id)

    shutil.rmtree("mlruns\.trash")
    os.mkdir("mlruns\.trash")


def init_experiments():
    clear_experiments()
    for name in [COMPLETE_EXPERIMENT, SIMPLIFIED_EXPERIMENT]:
        mlflow.create_experiment(name)
