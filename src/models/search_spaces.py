from hyperopt import hp
from hyperopt.pyll import scope

from src.models.available_models import ADABOOST, GRADIENT_BOOSTING, LINEAR_SVC, LOGISTIC_REG, RANDOM_FOREST, RIDGE, XGBOOST


# Linear SVC
# The combination of penalty='l1' and loss='hinge' is not supported.
L_SVC_SS = {
    "penalty": hp.choice("penalty", ["l1", "l2"]),
    "loss": hp.choice("loss", ["hinge", "squared_hinge"]),
    "C": hp.choice("C", [0.0001, 0.001, 0.01, 0.1, 1, 10]),
    "dual": hp.choice("dual", [True, False])
}

# Ridge
RIDGE_SS = {
    "alpha": hp.choice("alpha", [0.0001, 0.001, 0.01, 0.1, 1, 10]),
    "solver": hp.choice("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"])
}

# Logistic Regression
LR_SS = {
    "penalty": hp.choice("penalty", ['l1', 'l2', 'elasticnet', 'none']),
    "solver": hp.choice("solver", ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
    "C": hp.choice("C", [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]),
    "l1_ratio": hp.quniform("l1_ratio", 0, 1, 0.1),
}

# Gradient Boosting
GB_SS = {
    "n_estimators": hp.choice("n_estimators", [20, 50, 100, 300, 500, 1000]),
    "learning_rate": hp.choice("learning_rate", [0.0001, 0.005, 0.01, 0.05, 0.1, 0.2]),
    "max_depth": hp.quniform("max_depth", 8, 16, 1),
    "subsample": hp.choice("subsample", [0.7, 0.8, 0.9, 1]),
    "min_samples_split": hp.quniform("min_samples_split", 0.005, 0.02, 0.001),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 0.0001, 0.002, 0.0001)
}

# Ada Boost
ADA_SS = {
    "algorithm": hp.choice("algorithm", ["SAMME", "SAMME.R"]),
    "n_estimators": hp.choice("n_estimators", [10, 50, 100, 200, 500, 1000]),
    "learning_rate": hp.choice("learning_rate", [0.0001, 0.005, 0.01, 0.05, 0.1, 0.2])
}

# Random Forest
RF_SS = {
    "n_estimators": hp.choice("n_estimators", [300, 400, 500, 600]),
    "max_depth": hp.quniform("max_depth", 8, 16, 1),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "min_samples_split": hp.quniform("min_samples_split", 0.005, 0.02, 0.001),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 0.0001, 0.002, 0.0001)
}

# XGB
XGB_SS = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 200, 500, 1000]),
    "learning_rate": hp.choice("learning_rate", [0.001, 0.005, 0.01, 0.05]),
    "max_depth": scope.int(hp.quniform("max_depth", 5, 12, 1)),
    "min_child_weight": hp.quniform("min_child_weight", 1, 10, 2),
    "subsample": hp.choice("subsample", [0.7, 0.8, 0.9, 1]),
    "colsample_bytree": hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9]),
    "reg_alpha": hp.choice("reg_alpha", [0, 0.001, 0.005, 0.01, 0.05]),
    "gamma": hp.choice("gamma", [0, 0.1, 0.2, 0.3, 0.4])
}

SEARCH_SPACES = {
    LINEAR_SVC: L_SVC_SS,
    RIDGE: RIDGE_SS,
    LOGISTIC_REG: LR_SS,
    GRADIENT_BOOSTING: GB_SS,
    ADABOOST: ADA_SS,
    RANDOM_FOREST: RF_SS,
    XGBOOST: XGB_SS,
}
