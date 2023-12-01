import pytest
import numpy as np
from src.features.feature_selection import get_feature_importance  # Replace 'your_module' with the actual name of your module

# Mock data for testing
@pytest.fixture
def mock_data():
    X = np.random.default_rng(42).random((100, 5))
    y = np.random.default_rng(42).integers(2, size=100)
    return X, y

# Test the get_feature_importance function
def test_get_feature_importance_extra_trees(mock_data, capsys):
    X, y = mock_data
    feature_importance = get_feature_importance(X, y, verbose=True, classifier="ExtraTreesClassifier")

    # Check that the output is printed when verbose=True
    captured = capsys.readouterr()
    assert "Feature:" in captured.out
    assert "Score:" in captured.out

    # Check that the feature_importance array is returned
    assert isinstance(feature_importance, np.ndarray)
    assert len(feature_importance) == X.shape[1]

# Test the get_feature_importance function with RandomForestClassifier
def test_get_feature_importance_random_forest(mock_data, capsys):
    X, y = mock_data
    feature_importance = get_feature_importance(X, y, verbose=True, classifier="RandomForestClassifier")

    # Check that the output is printed when verbose=True
    captured = capsys.readouterr()
    assert "Feature:" in captured.out
    assert "Score:" in captured.out

    # Check that the feature_importance array is returned
    assert isinstance(feature_importance, np.ndarray)
    assert len(feature_importance) == X.shape[1]

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
