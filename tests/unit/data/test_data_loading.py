import os
import pytest
import pandas as pd
from sklearn.datasets import load_iris
from libmlops.data.data_loading import load_csv_data, save_datasets, load_datasets, clear_datasets


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

# Fixture to create temporary directories
@pytest.fixture
def temp_dirs(tmpdir):
    raw_dir = tmpdir.mkdir('data').mkdir('raw')
    return str(raw_dir)

@pytest.fixture
def cleanup_after_tests(temp_dirs):
    # Fixture to clean up files after tests
    yield
    clear_datasets(TEST_FILENAME, save_path=temp_dirs)


# Mock data for testing
TEST_DATASETS = [1, 2, 3]
TEST_FILENAMES = ['file1', 'file2', 'file3']
TEST_DATASET = [1]
TEST_FILENAME = 'file1'

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

def test_save_datasets(cleanup_after_tests, temp_dirs):
    save_datasets(TEST_DATASETS, TEST_FILENAMES, save_path=temp_dirs)

    # Check if files were created
    for filename in TEST_FILENAMES:
        dataset_path = os.path.join(temp_dirs, f"{filename}.dataset")
        assert os.path.exists(dataset_path)

def test_load_datasets(cleanup_after_tests, temp_dirs):
    # Save datasets before testing loading
    save_datasets(TEST_DATASETS, TEST_FILENAMES, save_path=temp_dirs)

    loaded_datasets = load_datasets(TEST_FILENAMES, save_path=temp_dirs)

    # Check if loaded datasets match the saved datasets
    assert loaded_datasets == TEST_DATASETS

def test_clear_datasets(cleanup_after_tests, temp_dirs):
    # Save datasets before testing clearing
    save_datasets(TEST_DATASETS, TEST_FILENAMES, save_path=temp_dirs)

    # Check if files exist before clearing
    for filename in TEST_FILENAMES:
        dataset_path = os.path.join(temp_dirs, f"{filename}.dataset")
        assert os.path.exists(dataset_path)

    clear_datasets(TEST_FILENAMES, save_path=temp_dirs)

    # Check if files were removed after clearing
    for filename in TEST_FILENAMES:
        dataset_path = os.path.join(temp_dirs, f"{filename}.dataset")
        assert not os.path.exists(dataset_path)

def test_save_dataset(cleanup_after_tests, temp_dirs):
    save_datasets(TEST_DATASET, TEST_FILENAME, save_path=temp_dirs)

    # Check if files were created
    filename = TEST_FILENAME
    dataset_path = os.path.join(temp_dirs, f"{filename}.dataset")
    assert os.path.exists(dataset_path)

def test_load_dataset(cleanup_after_tests, temp_dirs):
    # Save datasets before testing loading
    save_datasets(TEST_DATASET, TEST_FILENAME, save_path=temp_dirs)

    loaded_datasets = load_datasets(TEST_FILENAME, save_path=temp_dirs)

    # Check if loaded datasets match the saved datasets
    assert loaded_datasets == TEST_DATASET


def test_load_no_dataset(cleanup_after_tests, temp_dirs):
    # Delete existing datasets
    clear_datasets(TEST_FILENAME, save_path=temp_dirs)

    loaded_no_dataset = load_datasets(TEST_FILENAME, save_path=temp_dirs)

    # Check if loaded models match the saved models
    assert loaded_no_dataset  == []


def test_clear_datasets(cleanup_after_tests, temp_dirs):
    # Save datasets before testing clearing
    save_datasets(TEST_DATASET, TEST_FILENAME, save_path=temp_dirs)

    # Check if files exist before clearing
    filename = TEST_FILENAME
    dataset_path = os.path.join(temp_dirs, f"{filename}.dataset")
    assert os.path.exists(dataset_path)

    clear_datasets(TEST_FILENAME, save_path=temp_dirs)

    # Check if files were removed after clearing
    dataset_path = os.path.join(temp_dirs, f"{filename}.dataset")
    assert not os.path.exists(dataset_path)