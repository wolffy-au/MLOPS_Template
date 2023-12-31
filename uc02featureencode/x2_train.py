# 2. Model Training Script (train.py):
# This script trains machine learning models using preprocessed data.
# It includes model definition, hyperparameter tuning, and training.
# Executed during the CI/CD pipeline to update models with new data or retrain periodically.

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from libmlops.data.data_loading import load_datasets
from libmlops.models.model_loading import load_models, save_models
from libmlops.models.model_training import train_model
from libmlops.models.model_evaluation import (
    confusion_matrix_model,
    plot_confusion_matrix,
)

LOAD_MODEL = False


def run_train():
    print("Loading training datasets")
    [X_train, Y_train] = load_datasets(
        ["X_train", "Y_train"], "uc02featureencode/data/processed/"
    )

    model = []
    model_name = "finalised_model"
    if LOAD_MODEL:
        print("Loading previous model")
        model = load_models(model_name)

    if model == []:
        print("Does not exist - creating new model")
        model = KNeighborsClassifier()
        print("Training model")
        train_model(model, X_train, Y_train)
        print("Saving model")
        save_models(model, model_name, "uc02featureencode/data/processed/")

    pass


if __name__ == "__main__":
    run_train()
