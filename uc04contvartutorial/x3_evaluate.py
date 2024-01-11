# 3. Model Evaluation and Testing Script (evaluate.py):
# This script assesses the performance of trained models using validation or test datasets.
# It may generate metrics, visualizations, or reports for model evaluation.
# Typically executed after model training to ensure model quality before deployment.

from libmlops.data.data_loading import load_datasets
from libmlops.models.model_loading import load_models
from libmlops.utils.regressor_evaluation import model_evaluation


def run_evaluate():
    print("Loading validation datasets")
    [X_validation, Y_validation] = load_datasets(
        ["X_validation", "Y_validation"],
        save_path="uc04contvartutorial/data/processed/",
    )
    print("X_validation:\n", X_validation.head())
    print("Y_validation:\n", Y_validation.head())

    print("Loading previous model")
    model_name = "finalised_model"
    [model] = load_models(model_name, save_path="uc04contvartutorial/data/processed/")

    print("Evaluating model")
    r2, mae, cv_results_mean, cv_results_std = model_evaluation(
        model, X_validation, Y_validation
    )
    print("R-squared (coefficient of determination) regression score: ", r2, "\n")
    print("Mean Absolute Error:\n", mae)
    print("Cross-validation: %f Mean (%f Standard)" % (cv_results_mean, cv_results_std))

    pass


if __name__ == "__main__":
    run_evaluate()
