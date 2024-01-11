# 3. Model Evaluation and Testing Script (evaluate.py):
# This script assesses the performance of trained models using validation or test datasets.
# It may generate metrics, visualizations, or reports for model evaluation.
# Typically executed after model training to ensure model quality before deployment.

from libmlops.data.data_loading import load_datasets
from libmlops.models.model_loading import load_models
from libmlops.models.model_evaluation import (
    confusion_matrix_model,
    plot_confusion_matrix,
)
from libmlops.utils.classifier_evaluation import model_evaluation


def run_evaluate():
    print("Loading validation datasets")
    [X_validation, Y_validation] = load_datasets(
        ["X_validation", "Y_validation"], "uc02featureencode/data/processed/"
    )

    print("Loading previous model")
    model_name = "finalised_model"
    [model] = load_models(model_name, "uc02featureencode/data/processed/")

    print("Evaluating model")
    accuracy, report, cv_results_mean, cv_results_std = model_evaluation(
        model, X_validation, Y_validation
    )
    print("Accuracy score: ", accuracy, "\n")
    print("Classification report:\n", report)
    print("Cross-validation: %f Mean (%f Standard)" % (cv_results_mean, cv_results_std))

    cm = confusion_matrix_model(model, X_validation, Y_validation)
    print("Confusion Matrix:\n", cm)
    # plot_confusion_matrix(cm, model)
    pass


if __name__ == "__main__":
    run_evaluate()
