# Example model_evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Evaluation Metrics: This file typically contains functions for evaluating the performance of your trained models. These functions calculate metrics such as accuracy, precision, recall, F1 score, and others, depending on the nature of your problem (classification, regression, etc.).
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Visualization Functions: Functions for creating visualizations that help you understand model performance, such as confusion matrices, ROC curves, and precision-recall curves.
def plot_confusion_matrix(model, X_test, y_test):
    disp = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
    disp.ax_.set_title('Confusion Matrix')
    plt.show()

# Cross-Validation: If you use cross-validation to assess model performance, you might include functions for performing cross-validation and summarizing the results.
def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv)
    return scores.mean(), scores.std()
