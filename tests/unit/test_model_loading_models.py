import pytest
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from src.models.model_loading import save_models, load_models, clear_models

# Fixture to create temporary directories
@pytest.fixture
def temp_dirs(tmpdir):
    raw_dir = tmpdir.mkdir('data').mkdir('raw')
    return str(raw_dir)

# Mock data for testing
TEST_MODELS = [
    DecisionTreeClassifier(),
    GaussianNB(),
    SVC(gamma='auto')
    ]
TEST_FILENAMES = ['file1', 'file2', 'file3']

@pytest.fixture
def cleanup_after_tests(temp_dirs):
    # Fixture to clean up files after tests
    yield
    clear_models(TEST_FILENAMES, save_path=temp_dirs)

def test_save_models(cleanup_after_tests, temp_dirs):
    save_models(TEST_MODELS, TEST_FILENAMES, save_path=temp_dirs)

    # Check if files were created
    for filename in TEST_FILENAMES:
        model_path = os.path.join(temp_dirs, f"{filename}.model")
        assert os.path.exists(model_path)

def test_load_models(cleanup_after_tests, temp_dirs):
    # Save models before testing loading
    save_models(TEST_MODELS, TEST_FILENAMES, save_path=temp_dirs)

    loaded_models = load_models(TEST_FILENAMES, save_path=temp_dirs)

    # Check if loaded models match the saved models
    assert len(loaded_models) == len(TEST_MODELS)
    for loaded, original in zip(loaded_models, TEST_MODELS):
        assert loaded.get_params() == original.get_params()

def test_clear_models(cleanup_after_tests, temp_dirs):
    # Save models before testing clearing
    save_models(TEST_MODELS, TEST_FILENAMES, save_path=temp_dirs)

    # Check if files exist before clearing
    for filename in TEST_FILENAMES:
        model_path = os.path.join(temp_dirs, f"{filename}.model")
        assert os.path.exists(model_path)

    clear_models(TEST_FILENAMES, save_path=temp_dirs)

    # Check if files were removed after clearing
    for filename in TEST_FILENAMES:
        model_path = os.path.join(temp_dirs, f"{filename}.model")
        assert not os.path.exists(model_path)
