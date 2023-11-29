# 5. Inference Script (inference.py):
# This script is used for making predictions or inferences using a deployed model.
# It may include code for handling input data, calling the model, and processing the output.
# Part of the deployment pipeline to ensure that the deployed model functions as expected.

import pandas as pd

from data.data_loading import load_datasets
from models.model_loading import load_models
from models.model_evaluation import evaluate_model, cross_validate_model, confusion_matrix_model, plot_confusion_matrix

print("Loading datasets")
[X_train, Y_train] = load_datasets(["X_train", "Y_train"])

print("Loading previous model")
model_name = "finalised_model"
[model] = load_models(model_name)

print("Evaluating model")
accuracy, report = evaluate_model(model, X_train, Y_train)
print("Accuracy score: ", accuracy, "\n")
print("Classification report:\n", report)
