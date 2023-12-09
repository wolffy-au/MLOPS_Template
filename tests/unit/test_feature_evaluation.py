import pytest
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.features.feature_evaluation import get_feature_importance, get_k_best_features, get_recursive_feature_elimination, get_decision_tree, get_linear_regression

# Mock data for testing
@pytest.fixture
def mock_data():
    X = np.random.default_rng(42).random((100, 5))
    y = np.random.default_rng(42).integers(2, size=100)
    return X, y

# Normalise feature selection results
def normalise_feature_scores(results):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.array(results).reshape(-1, 1))
    return scaled_data.flatten()

# Test normalise_feature_scores function
def test_normalise_feature_scores():
    results = [0.1, 0.5, 0.8]
    normalized_results = normalise_feature_scores(results)
    assert np.all(normalized_results >= 0) and np.all(normalized_results <= 1)

# Test get_feature_importance function
def test_get_feature_importance(mock_data):
    X, y = mock_data
    feature_scores = get_feature_importance(X, y)

    # Check that the feature_scores array is returned
    assert isinstance(feature_scores, np.ndarray)
    assert len(feature_scores) == X.shape[1]
    assert np.all(feature_scores >= 0.)
    assert np.all(feature_scores <= 1.)

# Test get_k_best_features function
def test_get_k_best_features(mock_data):
    X, y = mock_data
    feature_scores = get_k_best_features(X, y, k=3)

    # Check that the feature_scores array is returned
    assert isinstance(feature_scores, np.ndarray)
    assert len(feature_scores) == X.shape[1]
    assert np.all(feature_scores >= 0.0)
    assert np.all(feature_scores <= 1.0)

# Test get_recursive_feature_elimination function
def test_get_recursive_feature_elimination(mock_data):
    X, y = mock_data
    feature_scores = get_recursive_feature_elimination(X, y)

    # Check that the feature_scores array is returned
    assert isinstance(feature_scores, np.ndarray)
    assert len(feature_scores) == X.shape[1]
    assert np.all(feature_scores >= 0.0)
    assert np.all(feature_scores <= 1.0)

# Test get_linear_regression function
def test_get_linear_regression(mock_data):
    X, y = mock_data
    feature_scores = get_linear_regression(X, y)

    # Check that the feature_scores array is returned
    assert isinstance(feature_scores, np.ndarray)
    assert len(feature_scores) == X.shape[1]
    assert np.all(feature_scores >= 0.0)
    assert np.all(feature_scores <= 1.0)

# Test get_decision_tree function
def test_get_decision_tree(mock_data):
    X, y = mock_data
    feature_scores = get_decision_tree(X, y)

    # Check that the feature_scores array is returned
    assert isinstance(feature_scores, np.ndarray)
    assert len(feature_scores) == X.shape[1]
    assert np.all(feature_scores >= 0.0)
    assert np.all(feature_scores <= 1.0)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
