# Load libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from libmlops.features.feature_evaluation import normalise_feature_scores

# Spot Check Feature Selection algorithms
models = [
    ('ExtraTreesClassifier', ExtraTreesClassifier()),
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

def features_evaluation(X_train, Y_train, verbose=False):
    # evaluate each model in turn
    features = []
    for name, model in models:
        model.fit(X_train, Y_train)
        # imp_results = model.feature_importances_
        # perform permutation importance
        imp_results = permutation_importance(model, X_train, Y_train, scoring='neg_mean_squared_error')
        imp_results_mean = normalise_feature_scores(imp_results['importances_mean'])
        # imp_results_std = normalise_feature_scores(imp_results['importances_std'])
        f = []
        for i,v in enumerate(imp_results_mean):
            if v >= 0.5:
                f.append(i)
                if i not in features:
                    features.append(i)

        if verbose:
            print(name, f)

    if verbose:
        print(features)
    
    return features

def keep_features(data, features, keep_y=False):
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
