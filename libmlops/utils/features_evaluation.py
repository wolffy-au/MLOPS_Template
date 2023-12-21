# Load libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Spot Check Feature Selection algorithms
models = [
    ('ExtraTreesClassifier', ExtraTreesClassifier()),
    ('HistGradientBoostingClassifier', HistGradientBoostingClassifier()),
    ('RandomForestClassifier', RandomForestClassifier()),
    ('RFE', RFE(LogisticRegression(solver='lbfgs', max_iter=1000))),
    ('LinearRegression', LinearRegression()),
    ('DecisionTreeRegressor', DecisionTreeRegressor()),
    ('RandomForestRegressor', RandomForestRegressor()),
    ('XGBRegressor', XGBRegressor()),
    ('KNeighborsRegressor', KNeighborsRegressor()),
    # ('PCA', PCA(n_components=4)),
    # ('SelectKBest', SelectKBest(score_func=f_classif, k="all")),
    ]

def keep_features(data, features, keep_y=False):
    # sort features highest to lowest
    # if all(isinstance(x, int) for x in features):
    features = sorted(features) #, reverse=True)

    if isinstance(data, pd.DataFrame):
        if keep_y:
            return data.iloc[:, (features + [-1])]
        else:
            return data.iloc[:, features]
    elif isinstance(data, np.ndarray):
        if keep_y:
            return data[:, (features + [-1])]
        else:
            return data[:, features]
    elif isinstance(data, list):
        if keep_y:
            return [[row[i] for i in (features + [-1])] for row in data]
        else:
            return [[row[i] for i in features] for row in data]
    else:
        raise ValueError("Input data type not supported. Use pandas DataFrame or numpy array.")

def compare_features(results, names):
    # Compare Features
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()
