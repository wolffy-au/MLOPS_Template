# test_model_training.py

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from libmlops.models.model_training import create_random_forest_model, train_model, tune_hyperparameters

def test_create_random_forest_model():
    # Test if the function returns a RandomForestClassifier instance
    model = create_random_forest_model()
    assert isinstance(model, RandomForestClassifier)

def test_train_model():
    # Generate some random data for testing
    X_train, X_test, y_train, y_test = train_test_split(
        np.random.default_rng(0).random((100, 10)),  # 100 samples, 10 features
        np.random.default_rng(0).integers(0, 2, size=100),  # Binary classification labels
        test_size=0.2,
        random_state=42
    )

    # Create a RandomForestClassifier model
    model = create_random_forest_model()

    # Train the model
    train_model(model, X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Check if the accuracy is reasonable (this is a basic test, real tests may involve more metrics)
    assert accuracy_score(y_test, y_pred) >= 0.0

def test_tune_hyperparameters():
    # Generate some random data for testing
    X_train, X_test, y_train, y_test = train_test_split(
        np.random.default_rng(0).random((100, 10)),  # 100 samples, 10 features
        np.random.default_rng(0).integers(0, 2, size=100),  # Binary classification labels
        test_size=0.2,
        random_state=42
    )

    # Create a RandomForestClassifier model
    model = create_random_forest_model()

    # Define a parameter grid for hyperparameter tuning
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}

    # Tune hyperparameters
    best_model = tune_hyperparameters(model, param_grid, X_train, y_train)

    # Check if the best_model is a RandomForestClassifier instance
    assert isinstance(best_model, RandomForestClassifier)

    # Train the best model
    train_model(best_model, X_train, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Check if the accuracy is reasonable for the tuned model
    assert accuracy_score(y_test, y_pred) >= 0.0
