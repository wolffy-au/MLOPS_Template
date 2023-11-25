import pytest
import pandas as pd
import os
import sklearn
from sklearn.datasets import load_iris
from src.data.data_loading import load_csv_data, save_datasets, load_datasets, clear_datasets

# Fixture to create temporary directories
@pytest.fixture
def temp_dirs(tmpdir):
    raw_dir = tmpdir.mkdir('data').mkdir('raw')
    return str(raw_dir)

# Create a sample CSV file for testing using the Iris dataset
@pytest.fixture
def sample_csv_file(tmp_path):
    # Load the Iris dataset from sklearn
    iris = load_iris()

    # Convert the Iris dataset to a DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Save the Iris dataset to a CSV file
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

# Test the load_csv_data function
def test_load_csv_data(sample_csv_file):
    # Call the function with the sample CSV file path
    result = load_csv_data(sample_csv_file)

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']
    assert all(column in result.columns for column in expected_columns)

    # Check if the DataFrame has rows
    assert len(result) > 0

# def test_load_database_data(connection_string, query):
#     # Code to connect to a database and fetch data
#     pass

# def test_load_api_data(api_endpoint):
#     # Code to make API requests and fetch data
#     pass

def test_save_and_load_datasets(temp_dirs):
    # Create some sample data
    datasets = [list(range(i, i+3)) for i in range(3)]
    filenames = ['dataset1', 'dataset2', 'dataset3']

    # Test save_datasets
    save_datasets(datasets, filenames, temp_dirs)

    # Test load_datasets
    loaded_datasets = load_datasets(filenames, temp_dirs)

    # Assertions
    assert len(loaded_datasets) == len(datasets)

    for loaded, original in zip(loaded_datasets, datasets):
        assert loaded == original

    # Clean up
    clear_datasets(filenames, temp_dirs)