# Example feature_selection.py

from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel, VarianceThreshold, SequentialFeatureSelector
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Feature Selection Functions: This file typically contains functions for selecting a subset of relevant features. This can help improve model performance and reduce overfitting.
def select_k_best_features(X, Y, score_func=f_classif, k=10, verbose=False):
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, Y)
    if verbose:
        # summarize scores
        print("X_selected:", X_selected)

    return X_selected

# Recursive Feature Elimination (RFE): Functions for recursively removing the least important features based on model performance.
def select_recursive_feature_elimination(X, Y, n_features_to_select=10, verbose=False):
    selector = RFE(LogisticRegression(solver='lbfgs', max_iter=1000), n_features_to_select=n_features_to_select)
    X_selected = selector.fit_transform(X, Y)
    if verbose:
        # summarize scores
        print("X_selected:", X_selected)

    return X_selected
