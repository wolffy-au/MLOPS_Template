import pytest
from unittest.mock import patch, MagicMock
from src.data.data_preprocessing import split_train_test

# Mocking the train_test_split function
@patch('src.data.data_preprocessing.train_test_split')  # Replace 'your_module' with the actual module name
def test_split_train_test(mock_train_test_split):
    # Mock data for testing
    mock_dataset = MagicMock()
    
    # Mock the train_test_split function to return predefined values
    mock_train_test_split.return_value = (
        [[1, 2], [3, 4], [5, 6]],  # X_train
        [[7, 8], [9, 10]],         # X_validation
        [0, 1, 0],                 # Y_train
        [1, 0]                     # Y_validation
    )

    # Call the function under test
    X_train, X_validation, Y_train, Y_validation = split_train_test(mock_dataset)

    # Assertions
    assert X_train == [[1, 2], [3, 4], [5, 6]]
    assert X_validation == [[7, 8], [9, 10]]
    assert Y_train == [0, 1, 0]
    assert Y_validation == [1, 0]

    # Ensure train_test_split was called with the correct arguments
    mock_train_test_split.assert_called_once_with(
        mock_dataset.values[:, :-1],  # X
        mock_dataset.values[:, -1],    # y
        test_size=0.2,                           # test_size
        random_state=42                             # random_state
    )

