import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from src.models.model_loading import save_models, load_models, clear_models

# Fixture to create temporary directories
@pytest.fixture
def temp_dirs(tmpdir):
    raw_dir = tmpdir.mkdir('data').mkdir('raw')
    return str(raw_dir)

def test_save_and_load_models(temp_dirs):
    # Create some sample data
    models = [
        DecisionTreeClassifier(),
        GaussianNB(),
        SVC(gamma='auto')
    ]

    filenames = ['model1', 'model2', 'model3']

    # Test save_models
    save_models(models, filenames, temp_dirs)

    # Test load_models
    loaded_models = load_models(filenames, temp_dirs)

    # Assertions
    assert len(loaded_models) == len(models)

    for loaded, original in zip(loaded_models, models):
        assert loaded.get_params() == original.get_params()

    # Clean up
    clear_models(filenames, temp_dirs)