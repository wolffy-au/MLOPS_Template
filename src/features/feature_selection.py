# Example feature_selection.py

from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Feature Selection Functions: This file typically contains functions for selecting a subset of relevant features. This can help improve model performance and reduce overfitting.
def select_k_best_features(X, y, k=10):
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected

# Recursive Feature Elimination (RFE): Functions for recursively removing the least important features based on model performance.
def recursive_feature_elimination(X, y, n_features_to_select=10):
    selector = RFE(model, n_features_to_select=n_features_to_select)
    X_selected = selector.fit_transform(X, y)
    return X_selected

# Feature Importance: Functions for obtaining feature importance scores from a trained model.
def get_feature_importance(X, y, verbose=False, classifier="ExtraTreesClassifier"):
    """
    Feature Importance: Functions for obtaining feature importance scores from a trained model.

    Parameters:
    - classifier (str): The setting for the meta estimator classifier. Default is "ExtraTreesClassifier".
      Possible settings include "ExtraTreesClassifier" or "RandomForestClassifier".
    """
    if classifier == "RandomForestClassifier":
        model = RandomForestClassifier()
    else:
        # Perform default actions
        model = ExtraTreesClassifier()

    model.fit(X, y)
    feature_importance = model.feature_importances_

    if verbose:
        # summarize feature importance
        for i,v in enumerate(feature_importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
    return feature_importance

# Show Feature Importance as a graph
def show_feature_importance(importance):
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()