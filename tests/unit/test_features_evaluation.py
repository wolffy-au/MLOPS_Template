import numpy as np
import pandas as pd

from libmlops.utils.features_evaluation import keep_features

# Test cases using pytest
def test_keep_features_pandas():
    # Create a pandas DataFrame
    df = pd.DataFrame({
        'Feature1': [1, 2, 3],
        'Feature2': [4, 5, 6],
        'Target': [7, 8, 9]
    })
    
    # Test keeping features and target column
    result_df = keep_features(df, features=[1, 0], keep_y=True)
    assert result_df.equals(df[['Feature1', 'Feature2', 'Target']])

    # Test keeping only specific features
    result_df = keep_features(df, features=[0], keep_y=False)
    assert result_df.equals(df[['Feature1']])


def test_keep_features_numpy():
    # Create a numpy array
    np_array = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
    
    # Test keeping features and target column
    result_np = keep_features(np_array, features=[2, 0, 1], keep_y=True)
    expected_result = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
    assert np.array_equal(result_np, expected_result)

    # Test keeping only specific features
    result_np = keep_features(np_array, features=[0, 2], keep_y=False)
    expected_result = np.array([[1, 7], [2, 8], [3, 9]])
    assert np.array_equal(result_np, expected_result)


# Run the tests
if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
