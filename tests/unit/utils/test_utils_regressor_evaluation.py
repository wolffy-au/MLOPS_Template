# test_feature_evaluation.py

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from libmlops.utils.regressor_evaluation import algorithm_evaluation, features_evaluation, compare_algorithms

def test_algorithm_evaluation():
    # Generate some random data for testing
    X_train, _, Y_train, _ = train_test_split(
        np.random.default_rng(0).random((100, 10)),  # 100 samples, 10 features
        np.random.default_rng(0).integers(0, 100, size=100),  # Regression target
        test_size=0.2,
        random_state=42
    )

    # Test the algorithm_evaluation function
    results, names = algorithm_evaluation(X_train, Y_train)

    # Check if the results and names have the correct structure
    assert isinstance(results, list)
    assert isinstance(names, list)

    # Check if each result is a list with two elements
    for result in results:
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

def test_features_evaluation():
    # Generate some random data for testing
    X_train, _, Y_train, _ = train_test_split(
        np.random.default_rng(0).random((100, 10)),  # 100 samples, 10 features
        np.random.default_rng(0).integers(0, 100, size=100),  # Regression target
        test_size=0.2,
        random_state=42
    )

    # Test the features_evaluation function
    features = features_evaluation(X_train, Y_train)

    # Check if the features result is a list
    assert isinstance(features, list)

    # Check if each feature is an integer
    for feature in features:
        assert isinstance(feature, int)
