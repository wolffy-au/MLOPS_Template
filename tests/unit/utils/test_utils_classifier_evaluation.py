import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from libmlops.models.model_evaluation import cross_validate_model
from libmlops.utils.classifier_evaluation import algorithm_evaluation, features_evaluation

# Sample data for testing
iris = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

def cross_validate_model(model, X_train, Y_train, cv, scoring):
    # Custom cross-validation function for testing purposes
    # Replace with the appropriate cross-validation method for your use case
    random_state = check_random_state(1)
    return cross_val_score(model, X_train, Y_train, cv=cv, scoring=scoring, random_state=random_state)

# def test_algorithm_evaluation():
#     # Test the algorithm_evaluation function
#     results, names = algorithm_evaluation(X_train, Y_train)
    
#     assert len(results) == len(names) == len(models)

#     for result, name, (_, model) in zip(results, names, models):
#         assert isinstance(result, list) and len(result) == 2
#         assert isinstance(result[0], float)
#         assert isinstance(result[1], float)
#         assert isinstance(name, str)

# def test_features_evaluation():
#     # Test the features_evaluation function
#     features = features_evaluation(X_train, Y_train)
    
#     assert isinstance(features, list)

#     for feature in features:
#         assert isinstance(feature, int)

# def test_normalise_feature_scores():
#     # Test the normalise_feature_scores function
#     scores = np.array([0.1, 0.2, 0.3])
#     normalized_scores = normalise_feature_scores(scores)

#     assert isinstance(normalized_scores, np.ndarray)
#     assert normalized_scores.shape == scores.shape
#     assert np.all(normalized_scores >= 0.0) and np.all(normalized_scores <= 1.0)

# Additional tests can be added based on specific cases and requirements.
