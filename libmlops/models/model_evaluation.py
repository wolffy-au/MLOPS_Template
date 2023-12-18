# Example model_evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Evaluation Metrics: This file typically contains functions for evaluating the performance of your trained models. These functions calculate metrics such as accuracy, precision, recall, F1 score, and others, depending on the nature of your problem (classification, regression, etc.).
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    return accuracy, report

# Cross-Validation: If you use cross-validation to assess model performance, you might include functions for performing cross-validation and summarizing the results.
def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores.mean(), scores.std()

# Performance Functions: Functions that help you understand model performance, such as confusion matrices, ROC curves, and precision-recall curves.
def confusion_matrix_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred, normalize='true')

# Visualization Functions: Functions for creating visualizations that help you understand model performance, such as confusion matrices, ROC curves, and precision-recall curves.
def plot_confusion_matrix(confusion_matrix, model):  
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title('Confusion Matrix')
    plt.show()

