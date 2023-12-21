# test_feature_selection.py

import numpy as np
from libmlops.features.feature_selection import (
    select_k_best_features,
    select_recursive_feature_elimination,
)

def test_select_k_best_features():
    # Create dummy data for testing
    X = np.random.rand(100, 20)  # 100 samples, 20 features
    Y = np.random.randint(0, 2, size=100)  # Binary classification labels

    # Test with default parameters
    X_selected = select_k_best_features(X, Y)

    # Check that the shape of the selected features is correct
    assert X_selected.shape == (X.shape[0], 10)  # Assuming default k=10

    # Test with custom parameters
    X_selected_custom = select_k_best_features(X, Y, score_func=lambda X, Y: np.ones(X.shape[1]), k=5)

    # Check that the shape of the selected features is correct for custom parameters
    assert X_selected_custom.shape == (X.shape[0], 5)

def test_select_recursive_feature_elimination():
    # Create dummy data for testing
    X = np.random.rand(100, 20)  # 100 samples, 20 features
    Y = np.random.randint(0, 2, size=100)  # Binary classification labels

    # Test with default parameters
    X_selected = select_recursive_feature_elimination(X, Y)

    # Check that the shape of the selected features is correct
    assert X_selected.shape == (X.shape[0], 10)  # Assuming default n_features_to_select=10

    # Test with custom parameters
    X_selected_custom = select_recursive_feature_elimination(X, Y, n_features_to_select=5)

    # Check that the shape of the selected features is correct for custom parameters
    assert X_selected_custom.shape == (X.shape[0], 5)
