import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from libmlops.models.model_evaluation import evaluate_classifier_model, evaluate_regressor_model, cross_validate_model, confusion_matrix_model

# Assuming 'your_module' is the module where your functions are defined

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def test_evaluate_classifier_model(classification_data):
    model, X_test, y_test = classification_data
    accuracy, report = evaluate_classifier_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
    assert isinstance(report, str)

def test_evaluate_regressor_model(regression_data):
    model, X_test, y_test = regression_data
    r2, mae = evaluate_regressor_model(model, X_test, y_test)
    assert isinstance(r2, float)
    assert isinstance(mae, float)

def test_cross_validate_model(classification_data):
    model, X_test, y_test = classification_data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    mean_score, std_score = cross_validate_model(model, X, y, cv=5, scoring='accuracy')
    assert isinstance(mean_score, float)
    assert isinstance(std_score, float)

def test_confusion_matrix_model(classification_data):
    model, X_test, y_test = classification_data
    confusion_matrix_result = confusion_matrix_model(model, X_test, y_test)
    assert isinstance(confusion_matrix_result, np.ndarray)
    assert confusion_matrix_result.shape == (2, 2)  # Modify shape based on your classes
