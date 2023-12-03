import pytest
import os
from src.data.data_loading import save_datasets, load_datasets, clear_datasets

# Fixture to create temporary directories
@pytest.fixture
def temp_dirs(tmpdir):
    raw_dir = tmpdir.mkdir('data').mkdir('raw')
    return str(raw_dir)

# Mock data for testing
TEST_DATASET = [1]
TEST_FILENAME = 'file1'

@pytest.fixture
def cleanup_after_tests(temp_dirs):
    # Fixture to clean up files after tests
    yield
    clear_datasets(TEST_FILENAME, save_path=temp_dirs)

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