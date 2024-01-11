# 2. Model Training Script (train.py):
# This script trains machine learning models using preprocessed data.
# It includes model definition, hyperparameter tuning, and training.
# Executed during the CI/CD pipeline to update models with new data or retrain periodically.

from sklearn.ensemble import RandomForestRegressor
from libmlops.data.data_loading import load_datasets
from libmlops.models.model_loading import load_models, save_models
from libmlops.models.model_training import train_model

LOAD_MODEL = True


def run_train():
    print("Loading training datasets")
    [X_train, Y_train] = load_datasets(
        ["X_train", "Y_train"], save_path="uc04contvartutorial/data/processed/"
    )

    model = []
    model_name = "finalised_model"
    if LOAD_MODEL:
        print("Loading previous model")
        model = load_models(model_name, save_path="uc04contvartutorial/data/processed/")

    if model == []:
        print("Does not exist - creating new model")
        model = RandomForestRegressor(n_jobs=-1)
        print("Training model")
        train_model(model, X_train, Y_train)
        print("Saving model")
        save_models(model, model_name, save_path="uc04contvartutorial/data/processed/")

    pass


if __name__ == "__main__":
    run_train()
