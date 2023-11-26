import pytest
import pandas as pd
from sklearn.datasets import load_iris
from src.data.data_loading import load_csv_data

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
