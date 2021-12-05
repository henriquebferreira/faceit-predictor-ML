
# Classifiers
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier

LINEAR_SVC = 'linear_svc'
RIDGE = 'ridge'
LOGISTIC_REG = 'logistic_regression'
GRADIENT_BOOSTING = 'gradient_boosting'
ADABOOST = 'adaboost'
RANDOM_FOREST = 'random_forest'
XGBOOST = 'xgboost'


def WrappedLinearSVC(*args, **kwargs):
    return CalibratedClassifierCV(LinearSVC(*args, **kwargs))


def WrappedRidgeClassifier(*args, **kwargs):
    return CalibratedClassifierCV(RidgeClassifier(*args, **kwargs))


MODELS = {
    LINEAR_SVC: WrappedLinearSVC,
    RIDGE: WrappedRidgeClassifier,
    LOGISTIC_REG: LogisticRegression,
    GRADIENT_BOOSTING: GradientBoostingClassifier,
    ADABOOST: AdaBoostClassifier,
    RANDOM_FOREST: RandomForestClassifier,
    XGBOOST: XGBClassifier,
}
