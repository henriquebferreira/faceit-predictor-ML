import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

SCORING_METRICS = ('accuracy', 'roc_auc')
CV_FOLDS = 5
SEED = 15

MODELS_PARAMS = {
    "Linear SVC": {
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "C":  [0.1, 1, 10, 100]
    },
    "Ridge":  {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    },
    "Linear Disc. Analysis": {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': np.arange(0, 1, 0.01)
    },
    "Logistic Regression": {
        "penalty": ['l1', 'l2', 'elasticnet', None],
        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        "C": [100, 10, 1.0, 0.1, 0.01]
    },
    "Gradient Boosting": {
        "n_estimators": [10, 100],
        "learning_rate": [0.01, 0.1],
        "min_samples_split": np.arange(0, 0.25, 0.03),
        "min_samples_leaf": np.arange(0, 0.25, 0.03),
        "max_depth": range(5, 16, 2),
        "max_features": range(7, 30, 3),
        "subsample": [0.7, 0.8, 0.9, 1]
    },
    "AdaBoost": {
        "n_estimators": [10, 50, 100, 200, 500, 1000],
        "learning_rate": [0.0001, 0.005, 0.01, 0.1, 1],
        "algorithm": ["SAMME", "SAMME.R"]
    },
    "Random Forest": {
        "criterion": ['gini', 'entropy'],
        "n_estimators": [20, 50, 100, 200, 500, 1000],
        "max_depth": [None, 5, 7, 9, 11, 13, 15],
        "max_features": ["sqrt", "log2", 5, 7, 9, 11, 13, 15],
        "min_samples_split": np.arange(0, 0.25, 0.03),
        "min_samples_leaf": np.arange(0, 0.25, 0.03),
    },
    "XGB": {
        "learning_rate": [0.001, 0.005, 0.01, 0.05],
        "n_estimators": [50, 100, 200, 500, 1000],
        'max_depth': range(5, 10, 2),
        'min_child_weight': range(1, 10, 2),
        "subsample": [0.7, 0.8, 0.9, 1],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4]
    }
}


def tune_hypparams(clf, params, X_train, y_train, method='randomized', cv=CV_FOLDS, scoring=SCORING_METRICS):
    if method == 'gridsearch':
        hyp_search = GridSearchCV(
            estimator=clf,
            param_grid=params,
            cv=cv,
            scoring=scoring,
            refit='roc_auc',
            n_jobs=-1,
            return_train_score=True,
            verbose=1)

    elif method == 'randomized':
        hyp_search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=params,
            n_iter=50,
            cv=cv,
            scoring=scoring,
            refit='roc_auc',
            n_jobs=-1,
            return_train_score=True,
            random_state=SEED,
            verbose=1)
    else:
        raise ValueError(
            "method must be one of the following: 'gridsearch', 'randomized'")

    hyp_search.fit(X_train, y_train)
    return hyp_search
