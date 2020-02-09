# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ---
# <p>&nbsp;</p>
# <div style="text-align:center"><span style="color:#0CA7DB; font-family:Play; font-size:3em;">FACEIT Predictor Notebook</span></div>
# <p>&nbsp;</p>
#
# <img style="float: center;" src="128.png">
#
# ---
#
# This notebook covers the development of the Machine Learning model to be used in the browser extension FACEIT Predictor. The model predicts the outcome of Counter Strike Global Offensive (a 5v5 First Person Shooter eSport) matches played on the FACEIT platform.

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Matplotlib-and-Seaborn-defaults" data-toc-modified-id="Matplotlib-and-Seaborn-defaults-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Matplotlib and Seaborn defaults</a></span></li><li><span><a href="#Load-Data" data-toc-modified-id="Load-Data-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Load Data</a></span></li></ul></li><li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Analyze-and-Describe" data-toc-modified-id="Analyze-and-Describe-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Analyze and Describe</a></span></li><li><span><a href="#Clean-Data" data-toc-modified-id="Clean-Data-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Clean Data</a></span></li><li><span><a href="#Players-Analysis" data-toc-modified-id="Players-Analysis-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Players Analysis</a></span></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#New-Experimental-Features" data-toc-modified-id="New-Experimental-Features-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>New Experimental Features</a></span></li></ul></li><li><span><a href="#Visualization" data-toc-modified-id="Visualization-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Visualization</a></span></li><li><span><a href="#Prepare-data-for-training" data-toc-modified-id="Prepare-data-for-training-2.6"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Prepare data for training</a></span></li></ul></li><li><span><a href="#Baseline" data-toc-modified-id="Baseline-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Baseline</a></span><ul class="toc-item"><li><span><a href="#Model" data-toc-modified-id="Model-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Model</a></span></li><li><span><a href="#Evaluation" data-toc-modified-id="Evaluation-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Evaluation</a></span></li><li><span><a href="#Visualization" data-toc-modified-id="Visualization-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Visualization</a></span></li></ul></li><li><span><a href="#Model-and-Feature-Selection" data-toc-modified-id="Model-and-Feature-Selection-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Model and Feature Selection</a></span></li><li><span><a href="#Model-Deployment" data-toc-modified-id="Model-Deployment-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Model Deployment</a></span><ul class="toc-item"><li><span><a href="#Save-Model" data-toc-modified-id="Save-Model-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Save Model</a></span></li></ul></li></ul></div>
# -

# # Introduction
# TODO: WRITE THIS SECTION State the purpose of the notebook here and how it is structured

# ## Imports
# Import libraries and other required jupyter notebooks and python modules.

# +
# Data manipulation
import pandas as pd
import numpy as np

# Options for pandas
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30

# Utils for feature creation
from datetime import datetime
import math
import scipy
from statistics import mean
from sklearn.preprocessing import StandardScaler

# Feature selection
from sklearn.feature_selection import RFE, RFECV

# Outlier Detection
from sklearn.neighbors import LocalOutlierFactor

# Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, 
                              ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier

# Neural network libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Model selection and hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV

# Classifier metrics
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import permutation_importance

# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Save the ML model
import joblib

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from IPython import get_ipython
ipython = get_ipython()

# Developed python modules
import load_faceit_data as loader
import add_features
# Import other jupyter notebooks
#import import_ipynb
#import MongoDBAtlas

# autoreload extension
if 'autoreload' not in ipython.extension_manager.loaded:
    %load_ext autoreload

# %autoreload 2
# %autosave 0
# -

# ## Matplotlib and Seaborn defaults

sns.set_style("dark")
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['text.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelsize'] = '13'
plt.rcParams['axes.titlesize'] = '20'

# ## Load Data
# Currently, the data can be loaded in two ways:
# 1. From a locally stored JSON file
# 2. From a MongoDB database hosted in the local network

params = {'filename':'data\matches_02.json'}
dataset = loader.load_data(load_type='json', **params) 

# # Data Preprocessing

# ## Analyze and Describe
# TODO: Add more functions to pre-analyze the data

dataset.describe()

# ## Clean Data

faceit_maps = ['de_train', 'de_inferno', 'de_mirage', 'de_vertigo', 'de_nuke', 'de_overpass', 'de_cache', 'de_dust2']
date_format = "%Y-%m-%dT%H:%M:%SZ"


# +
def clean_dataframe(data):
    print("Dataframe's shape before cleaning", data.shape)
    
    data.drop(columns=['_id', '_id__stitch_transaction', 'state', 'match_status'], inplace=True)
    data = data[data['score'].notnull()]
    data = data[data['mapPlayed'].isin(faceit_maps)]
    
    # Removal of duplicate matches
    data.drop_duplicates(subset=['match_id'],keep="first", inplace=True)
    
    # Removal of non 5v5 matches
    data.loc[:,'num_players'] = data.apply(lambda row: get_num_players(row), axis=1).values
    data = data[data['num_players'] == 10]
    
    print("Dataframe's shape after cleaning", data.shape)
    return data

def get_num_players(row):
    return len(row['teamA']) +len(row['teamB'])


# -

dataset = clean_dataframe(dataset)


# ## Players Analysis

# +
def get_all_players_elos(data):
    all_players = []
    for _, row in data.iterrows():
        for player_id,player_info in row['teamA'].items():
            all_players.append(player_info['elo'])
        for player_id,player_info in row['teamB'].items():
            all_players.append(player_info['elo'])
    return all_players

def get_player_elo_kde(data):
    player_elos = get_all_players_elos(data)
    player_elos = np.array(player_elos) 
    player_elos_series = pd.Series(player_elos)
    kde = player_elos_series.plot.kde(ind=1000)
    xdata, ydata = kde.get_lines()[0].get_data()
    return scipy.integrate.cumtrapz(ydata, xdata, dx=1, initial=0), xdata

def get_elo_dif_prob(lower_bound, upper_bound, cdf, bins):
    lower_bin = (np.abs(bins-lower_bound)).argmin()
    upper_bin = (np.abs(bins-upper_bound)).argmin()
    return cdf[upper_bin] - cdf[lower_bin]

players_elo_distribution, elo_bins = get_player_elo_kde(dataset)


# -

def get_all_players_info(data):
    players_id = []
    elos = []
    matches = []
    winrates = []
    kds = []
    hs_percents = []
    #add createdAt (accounts created a long time ago with few matches might be smurfs)
    for _, row in data.iterrows():
        for player_id,player_info in row['teamA'].items():
            if (player_info['lifetimeData']== None):
                players_id.append(0)
                elos.append(0)
                matches.append(0)
                winrates.append(0)
                kds.append(0)
                hs_percents.append(0)
                continue
            players_id.append(player_info['id'])
            elos.append(player_info['elo'])
            matches.append(int(player_info['lifetimeData']['matches']))
            winrates.append(int(player_info['lifetimeData']['winRate']))
            kds.append(float(player_info['lifetimeData']['averageKD']))
            hs_percents.append(int(player_info['lifetimeData']['averageHS']))
            
        for player_id,player_info in row['teamB'].items():
            if (player_info['lifetimeData']== None):
                players_id.append(0)
                elos.append(0)
                matches.append(0)
                winrates.append(0)
                kds.append(0)
                hs_percents.append(0)
                continue
            players_id.append(player_info['id'])
            elos.append(player_info['elo'])
            matches.append(int(player_info['lifetimeData']['matches']))
            winrates.append(int(player_info['lifetimeData']['winRate']))
            kds.append(float(player_info['lifetimeData']['averageKD']))
            hs_percents.append(int(player_info['lifetimeData']['averageHS']))
            
    return np.array(players_id), np.array(elos), np.array(matches), np.array(winrates), np.array(kds), np.array(hs_percents)


players_id, elos, matches, winrates, kds, hs_percents = get_all_players_info(dataset)
array = [[elos[i], matches[i], winrates[i], kds[i]] for i in range(len(elos))]
X = np.array(array)

mean_winrate = mean(winrates)
mean_kd = mean(kds)

lof = LocalOutlierFactor(contamination=0.05)
lof.fit(X)


def smurf_or_cheater_prob(outlier_model):
    # Check if user is verified (then, it is probably a pro)
    transformed_outlier_factor = []
    for i in range(len(outlier_model.negative_outlier_factor_)):
        if (outlier_model.negative_outlier_factor_[i]< -1) and (winrates[i]>mean_winrate) and (kds[i]> mean_kd):
            transformed_outlier_factor.append(math.log(-outlier_model.negative_outlier_factor_[i])*100)
        else:
            transformed_outlier_factor.append(0)
        arr = np.array(transformed_outlier_factor)     
        
    team_A = arr.reshape(-1,5)[::2,:]
    team_B = arr.reshape(-1,5)[1::2,:]

    dataset['smurf_or_cheater_A'] = np.mean(team_A, axis=1)
    dataset['smurf_or_cheater_B'] = np.mean(team_B, axis=1)
    dataset['dif_smurf_or_cheater'] = dataset['smurf_or_cheater_A'] - dataset['smurf_or_cheater_B']


smurf_or_cheater_prob(lof)

# ## Feature Engineering

dataset.loc[:, 'unix_start_time'] = pd.to_datetime(dataset['startTime'], format=date_format).values.astype('datetime64[s]').astype('int')
dataset.drop(columns=['startTime'], inplace=True)

add_features.add_all_team_features(dataset)

dataset.loc[:, 'missing_info'] = dataset.apply(lambda row: add_features.get_missing_info(row), axis=1).values


# +
def dif_elo_prob(row):
    return get_elo_dif_prob(row['mean_elo_A'], row['mean_elo_B'], players_elo_distribution, elo_bins)

dataset.loc[:,'dif_elo_prob'] = dataset.apply(lambda row: dif_elo_prob(row), axis=1).values


# +
def convert_winner_to_numeric(row):
    winner_numeric = 0 if row['score'] == 'faction1' else 1
    return winner_numeric

dataset.loc[:, 'winner'] = dataset.apply(lambda row: convert_winner_to_numeric(row), axis=1).values


# -

# ### New Experimental Features
# Creation and test of new features. Once validated the correspondent function should be moved to `add_features.py`.

# ## Visualization

# +
def split_dataset_winners(dataset):
    return dataset[dataset['winner']==0], dataset[dataset['winner']==1]

data_winner_A, data_winner_B = split_dataset_winners(dataset)


# -

def comp_featured_based_on_winner(feature, num_bins=100, title=None):
    plt.figure(figsize=(12,6))
    ax = sns.distplot(data_winner_A[feature],
                      kde=False,
                      bins=num_bins,
                      color='#FF5500',
                      hist_kws=dict(alpha=0.8))
    print('Team A - Feature Mean Value',np.mean(data_winner_A[feature]))
    
    sns.distplot(data_winner_B[feature],
                 kde=False,
                 bins=num_bins,
                 color='#141616',
                 hist_kws=dict(alpha=0.8),
                 ax = ax)
    print('Team B - Feature Mean Value',np.mean(data_winner_B[feature]))
    ax.patch.set_alpha(0.1)
    ax.set_title(title)
    ax.legend(['A', 'B'], title="Winner team")
    return ax


# +
ax = comp_featured_based_on_winner('dif_mean_winrate_preference', num_bins=30)

# Configure axes limits
# ax.set_xlim(x_left, x_right)
# ax.set_ylim(y_bottom, y_top)
# -

# ## Prepare data for training

# +
selected_cols = ['dif_new_players',
                'dif_mean_matches',
                'dif_mean_matches_on_map',
                'dif_mean_winrate_on_map',
                'dif_mean_kd_on_map',
                'dif_mean_matches_preference',
                'dif_mean_winrate_preference',
                'dif_mean_kd_preference',
                'dif_mean_elo',
                'dif_stddev_elo',
                'dif_paid_memberships',
                'dif_solo_players',
                'dif_num_parties',
                'dif_mean_matches_on_map_prev',
                'dif_mean_winrate_prev',
                'dif_multikills_prev',
                'dif_mean_assists_prev',
                'dif_mean_kd_prev',
                'dif_mean_time_prev',
                'dif_delta_mean_elo_prev',
                'dif_smurf_or_cheater',
                'dif_max_time_prev',
                'dif_elo_prob',
                'dif_first_match',
                'dif_mean_time_created_at',
                'dif_stddev_time_created_at',
                'dif_min_time_created_at',
                'dif_mean_matches_today',
                'dif_played_map_today',
                'dif_have_played_together_prev',
                'winner']

data_processed = dataset[selected_cols]
data_label = dataset['winner']
data_features = dataset.drop(columns=['winner'])
# -

# # Baseline
# Show graphs and stats here

# ## Model

X_train, X_test, y_train, y_test = train_test_split(data_features,
                                                    data_label,
                                                    test_size=0.3,
                                                    random_state=42)

# Baseline model: Random Forest with default parameters
rf = RandomForestClassifier()

# Alternative Baseline Model with slightly better performance
    # rf = RandomForestClassifier(n_estimators=1500,
    #                             max_features= 0.3,
    #                             max_depth=7,
    #                             min_samples_leaf=0.005,
    #                             random_state=41)

rf.fit(X_train, y_train)


# ## Evaluation

def print_metrics(model, X_train, y_train, X_test, y_test):
    print("Model Score (Mean accuracy on test data)", model.score(X_test, y_test))
    pred = model.predict(X_test)
    print("\nClassification Report")
    print(classification_report(y_test, pred))
    roc_graph = plot_roc_curve(model, X_test, y_test)
    roc_graph.ax_.patch.set_alpha(0.1)


print_metrics(rf, X_train, y_train, X_test, y_test)

# Displays the correlation matrix regarding the features present in dataset
corr_mat = data_processed.corr()
plt.figure(figsize=(20,20))
ax = sns.heatmap(corr_mat, annot=True, cbar=False, annot_kws={"size": 10})
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# +
permutation_info = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=0, n_jobs=-1)

features_info = list(zip(X_train.columns,
                         permutation_info['importances_mean'],
                         permutation_info['importances_std']))

# Sort by descending mean feature importance
features_info = sorted(features_info, key=lambda feature: feature[1], reverse=True)

features_imp_df = pd.DataFrame(features_info, columns =['Feature_Name',
                                                        'Mean_Importance',
                                                        'StdDev_Importance'])

features_imp_df.head(40)
# -

# ## Visualization



# # Model and Feature Selection

# Create train and test set  
X_train, X_test, y_train, y_test = train_test_split(data_features,
                                                    data_label,
                                                    test_size=0.3,
                                                    random_state=42)

# +
# Classifiers
classifiers = {}
classifiers.update({"LDA": LinearDiscriminantAnalysis()})
classifiers.update({"QDA": QuadraticDiscriminantAnalysis()})
classifiers.update({"AdaBoost": AdaBoostClassifier()})
classifiers.update({"Bagging": BaggingClassifier()})
classifiers.update({"Extra Trees Ensemble": ExtraTreesClassifier()})
classifiers.update({"Gradient Boosting": GradientBoostingClassifier()})
classifiers.update({"Random Forest": RandomForestClassifier()})
classifiers.update({"Ridge": RidgeClassifier()})
classifiers.update({"SGD": SGDClassifier()})
classifiers.update({"BNB": BernoulliNB()})
classifiers.update({"GNB": GaussianNB()})
classifiers.update({"KNN": KNeighborsClassifier()})
classifiers.update({"MLP": MLPClassifier()})
classifiers.update({"LSVC": LinearSVC()})
classifiers.update({"NuSVC": NuSVC()})
classifiers.update({"SVC": SVC()})
classifiers.update({"DTC": DecisionTreeClassifier()})
classifiers.update({"ETC": ExtraTreeClassifier()})

# Create dict of decision function labels
DECISION_FUNCTIONS = {"Ridge", "SGD", "LSVC", "NuSVC", "SVC"}

# Create dict for classifiers with feature_importances_ attribute
FEATURE_IMPORTANCE = {"Gradient Boosting", "Extra Trees Ensemble", "Random Forest"}

# +
# Hyperparameter configuration

# Initiate parameter grid
parameters = {}

# Update dict with LDA
parameters.update({"LDA": {"classifier__solver": ["svd"], 
                                         }})

# Update dict with QDA
parameters.update({"QDA": {"classifier__reg_param":[0.01*ii for ii in range(0, 101)], 
                                         }})
# Update dict with AdaBoost
parameters.update({"AdaBoost": { 
                                "classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                "classifier__n_estimators": [200],
                                "classifier__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]
                                 }})

# Update dict with Bagging
parameters.update({"Bagging": { 
                                "classifier__base_estimator": [DecisionTreeClassifier(max_depth = ii) for ii in range(1,6)],
                                "classifier__n_estimators": [200],
                                "classifier__max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                "classifier__n_jobs": [-1]
                                }})

# Update dict with Gradient Boosting
parameters.update({"Gradient Boosting": { 
                                        "classifier__learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001], 
                                        "classifier__n_estimators": [200],
                                        "classifier__max_depth": [2,3,4,5,6],
                                        "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                        "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                        "classifier__max_features": ["auto", "sqrt", "log2"],
                                        "classifier__subsample": [0.8, 0.9, 1]
                                         }})


# Update dict with Extra Trees
parameters.update({"Extra Trees Ensemble": { 
                                            "classifier__n_estimators": [200],
                                            "classifier__class_weight": [None, "balanced"],
                                            "classifier__max_features": ["auto", "sqrt", "log2"],
                                            "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__criterion" :["gini", "entropy"]     ,
                                            "classifier__n_jobs": [-1]
                                             }})


# Update dict with Random Forest Parameters
parameters.update({"Random Forest": { 
                                    "classifier__n_estimators": [400],
                                    "classifier__class_weight": ["balanced"],
                                    "classifier__max_features": ["auto", "sqrt"],
                                    "classifier__max_depth" : [10,11,12],
                                    "classifier__min_samples_split": [0.001],
                                    "classifier__min_samples_leaf": [0.001],
                                    "classifier__criterion" :["gini", "entropy"]     ,
                                    "classifier__n_jobs": [-1]
                                     }})


# Update dict with Ridge
parameters.update({"Ridge": { 
                            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
                             }})

# Update dict with SGD Classifier
parameters.update({"SGD": { 
                            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
                            "classifier__penalty": ["l1", "l2"],
                            "classifier__n_jobs": [-1]
                             }})


# Update dict with BernoulliNB Classifier
parameters.update({"BNB": { 
                            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
                             }})

# Update dict with GaussianNB Classifier
parameters.update({"GNB": { 
                            "classifier__var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5]
                             }})

# Update dict with K Nearest Neighbors Classifier
parameters.update({"KNN": { 
                            "classifier__n_neighbors": list(range(1,31)),
                            "classifier__p": [1, 2, 3, 4, 5],
                            "classifier__leaf_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                            "classifier__n_jobs": [-1]
                             }})

# Update dict with MLPClassifier
parameters.update({"MLP": { 
                            "classifier__hidden_layer_sizes": [(5), (10), (5,5), (10,10), (5,5,5), (10,10,10)],
                            "classifier__activation": ["identity", "logistic", "tanh", "relu"],
                            "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
                            "classifier__max_iter": [100, 200, 300, 500, 1000, 2000],
                            "classifier__alpha": list(10.0 ** -np.arange(1, 10)),
                             }})

parameters.update({"LSVC": { 
                            "classifier__penalty": ["l2"],
                            "classifier__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]
                             }})

parameters.update({"NuSVC": { 
                            "classifier__nu": [0.25, 0.50, 0.75],
                            "classifier__kernel": ["linear", "rbf", "poly"],
                            "classifier__degree": [1,2,3,4,5,6],
                             }})

parameters.update({"SVC": { 
                            "classifier__kernel": ["linear", "rbf", "poly"],
                            "classifier__gamma": ["auto"],
                            "classifier__C": [0.1, 0.5, 1, 5, 10, 50, 100],
                            "classifier__degree": [1, 2, 3, 4, 5, 6]
                             }})


# Update dict with Decision Tree Classifier
parameters.update({"DTC": { 
                            "classifier__criterion" :["gini", "entropy"],
                            "classifier__splitter": ["best", "random"],
                            "classifier__class_weight": [None, "balanced"],
                            "classifier__max_features": ["auto", "sqrt", "log2"],
                            "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})

# Update dict with Extra Tree Classifier
parameters.update({"ETC": { 
                            "classifier__criterion" :["gini", "entropy"],
                            "classifier__splitter": ["best", "random"],
                            "classifier__class_weight": [None, "balanced"],
                            "classifier__max_features": ["auto", "sqrt", "log2"],
                            "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})

# +
# Define classifier to use as the base of the recursive feature elimination algorithm
selected_classifier = "Random Forest"
classifier = classifiers[selected_classifier]

# Tune classifier (Took = 4.8 minutes)
    
# Scale features via Z-score normalization
scaler = StandardScaler()

# Define steps in pipeline
steps = [("scaler", scaler), ("classifier", classifier)]

# Initialize Pipeline object
pipeline = Pipeline(steps = steps)
  
# Define parameter grid
param_grid = parameters[selected_classifier]

# Initialize GridSearch object
gscv = GridSearchCV(pipeline, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = "roc_auc")
                  
# Fit gscv
print(f"Now tuning {selected_classifier}. Go grab a beer or something.")
gscv.fit(X_train, np.ravel(y_train))  

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_
        
# Update classifier parameters
tuned_params = {item[12:]: best_params[item] for item in best_params}
classifier.set_params(**tuned_params)


# -

# Select Features using RFECV
class PipelineRFE(Pipeline):
    # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self


# +
# Define pipeline for RFECV
steps = [("scaler", scaler), ("classifier", classifier)]
pipe = PipelineRFE(steps = steps)

# Initialize RFECV object
feature_selector = RFECV(pipe, cv = 5, step = 1, scoring = "roc_auc", verbose = 1)

# Fit RFECV
feature_selector.fit(X_train, np.ravel(y_train))

# Get selected features
feature_names = X_train.columns
selected_features = feature_names[feature_selector.support_].tolist()

# +
# Get Performance Data
performance_curve = {"Number of Features": list(range(1, len(feature_names) + 1)),
                    "AUC": feature_selector.grid_scores_}
performance_curve = pd.DataFrame(performance_curve)

# Performance vs Number of Features
# Set graph style
sns.set(font_scale = 1.75)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})
colors = sns.color_palette("RdYlGn", 20)
line_color = colors[3]
marker_colors = colors[-1]

# Plot
f, ax = plt.subplots(figsize=(13, 6.5))
sns.lineplot(x = "Number of Features", y = "AUC", data = performance_curve,
             color = line_color, lw = 4, ax = ax)
sns.regplot(x = performance_curve["Number of Features"], y = performance_curve["AUC"],
            color = marker_colors, fit_reg = False, scatter_kws = {"s": 200}, ax = ax)

# Axes limits
plt.xlim(0.5, len(feature_names)+0.5)
plt.ylim(0.60, 0.925)

# Generate a bolded horizontal line at y = 0
ax.axhline(y = 0.625, color = 'black', linewidth = 1.3, alpha = .7)

# Turn frame off
ax.set_frame_on(False)

# Tight layout
plt.tight_layout()
# -

performance_curve.head(40)

# +
# Define pipeline for RFECV
steps = [("scaler", scaler), ("classifier", classifier)]
pipe = PipelineRFE(steps = steps)

# Initialize RFE object
feature_selector = RFE(pipe, n_features_to_select =21, step = 1, verbose = 1)

# Fit RFE
feature_selector.fit(X_train, np.ravel(y_train))

# Get selected features labels
feature_names = X_train.columns
selected_features = feature_names[feature_selector.support_].tolist()

# +
# Get selected features data set
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# Train classifier
classifier.fit(X_train, np.ravel(y_train))

# Get feature importance
feature_importance = pd.DataFrame(selected_features, columns = ["Feature Label"])
feature_importance["Feature Importance"] = classifier.feature_importances_

# Sort by feature importance
feature_importance = feature_importance.sort_values(by="Feature Importance", ascending=False)

# Set graph style
sns.set(font_scale = 1.75)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})

# Set figure size and create barplot
f, ax = plt.subplots(figsize=(12, 9))
sns.barplot(x = "Feature Importance", y = "Feature Label",
            palette = reversed(sns.color_palette('YlOrRd', 15)),  data = feature_importance)

# Generate a bolded horizontal line at y = 0
ax.axvline(x = 0, color = 'black', linewidth = 4, alpha = .7)

# Turn frame off
ax.set_frame_on(False)

# Tight layout
plt.tight_layout()
# -

# # Model Deployment

# ## Save Model

# +
# TODO: model version control and auto-deployment

# Save the model as a pickle in a file 
joblib.dump(model, 'model.pkl') 
# -


