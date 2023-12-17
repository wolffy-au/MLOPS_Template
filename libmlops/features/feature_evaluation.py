# Example feature_evaluation.py

import numpy as np
from matplotlib import pyplot
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel, VarianceThreshold, SequentialFeatureSelector
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Normalise feature selection results
def normalise_feature_scores(results):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.array(results).reshape(-1, 1))
    return scaled_data.flatten()

# Feature Importance: Functions for obtaining feature importance scores from a trained model.
def get_feature_importance(X, Y, verbose=False):
    model=ExtraTreesClassifier()
    model.fit(X, Y)
    feature_importance = model.feature_importances_
    if verbose:
        # summarize feature importance
        for i,v in enumerate(feature_importance):
            print('Feature: %0d, Score: %.5f' % (i,v))

    return normalise_feature_scores(feature_importance)

# Feature Selection Functions: This file typically contains functions for selecting a subset of relevant features. This can help improve model performance and reduce overfitting.
def get_k_best_features(X, Y, k='all', verbose=False):
    selector = SelectKBest(score_func=f_classif, k=k)
    fit = selector.fit(X, Y)
    if verbose:
        # summarize scores
        set_printoptions(precision=2)
        print("Fit scores:", fit.scores_)

    return normalise_feature_scores(fit.scores_)

# Recursive Feature Elimination (RFE): Functions for recursively removing the least important features based on model performance.
def get_recursive_feature_elimination(X, Y, verbose=False):
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    selector = RFE(model, n_features_to_select=1)
    fit = selector.fit(X, Y)
    if verbose:
        # summarize scores
        set_printoptions(precision=2)
        print("Fit ranking:", fit.ranking_)

    return normalise_feature_scores(fit.ranking_)

# Recursive Feature Elimination (RFE): Functions for recursively removing the least important features based on model performance.
def get_recursive_feature_elimination(X, Y, verbose=False):
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    selector = RFE(model, n_features_to_select=1)
    fit = selector.fit(X, Y)
    if verbose:
        # summarize scores
        set_printoptions(precision=2)
        print("Fit ranking:", fit.ranking_)

    return normalise_feature_scores(fit.ranking_)

# Linear Regression: Functions for recursively removing the least important features based on model performance.
def get_linear_regression(X, Y, verbose=False):
    model = LinearRegression()
    model.fit(X, Y)
    if verbose:
        # summarize scores
        set_printoptions(precision=2)
        print("Feature coefficient:", model.coef_)

    return normalise_feature_scores(model.coef_)

# Linear Regression: Functions for recursively removing the least important features based on model performance.
def get_decision_tree(X, Y, verbose=False):
    model = DecisionTreeRegressor()
    model.fit(X, Y)
    if verbose:
        # summarize scores
        set_printoptions(precision=2)
        print("Feature importances:", model.feature_importances_)

    return normalise_feature_scores(model.feature_importances_)

# Show Feature Importance as a graph
def show_feature_importance(importance):
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()