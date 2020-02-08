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
# <font color="#0CA7DB" size="+4">FACEIT Predictor Notebook</font>
# <p>&nbsp;</p>
#
# ---
#
# This notebook covers the development of the Machine Learning model to be used in the browser extension FACEIT Predictor. The model predicts the outcome of Counter Strike Global Offensive (a 5v5 First Person Shooter eSport) matches played on the FACEIT platform.

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1.0.1"><span class="toc-item-num">1.0.1&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Load-Data" data-toc-modified-id="Load-Data-1.0.2"><span class="toc-item-num">1.0.2&nbsp;&nbsp;</span>Load Data</a></span></li></ul></li></ul></li><li><span><a href="#Analysis/Modeling" data-toc-modified-id="Analysis/Modeling-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Analysis/Modeling</a></span></li><li><span><a href="#Results" data-toc-modified-id="Results-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Results</a></span></li><li><span><a href="#Conclusions-and-Next-Steps" data-toc-modified-id="Conclusions-and-Next-Steps-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Conclusions and Next Steps</a></span></li></ul></div>
# -

# # Introduction
# TODO: WRITE THIS SECTION State the purpose of the notebook here and how it is structured

# + [markdown] heading_collapsed=true
# ### Imports
# Import libraries and other required jupyter notebooks.

# + hidden=true
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
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import StackingClassifier

# Neural network libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Model selection and hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV

# Classifier metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import plot_roc_curve

# Visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from IPython import get_ipython
ipython = get_ipython()

# autoreload extension
if 'autoreload' not in ipython.extension_manager.loaded:
    %load_ext autoreload

# %autoreload 2
# %autosave 0
# -

# ### Load Data
# At the moment, data can be loaded in two fashions:
# 1. From a locally stored JSON file
# 2. From a MongoDB database hosted in the local network

# +
# Python program to illustrate 
from os import path

def load_data(load_type, **kwargs): 
    if(load_type == 'json'):
        file_path = kwargs.get('filename', None)
        if not path.exists(file_path):
            raise IOError("File {} doesn't exist".format(file_path))
        return pd.read_json(file_path, lines=True)
    elif(load_type == 'mongoDB'):
        return
        
params = {'filename':'data\matches_05.json'}
dataset = load_data(load_type='json', **params) 
# -

# # Analysis/Modeling
# Do work here

# # Results
# Show graphs and stats here

# # Conclusions and Next Steps
# Summarize findings here


