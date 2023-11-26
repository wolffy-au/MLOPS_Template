# 2. Model Training Script (train.py):
# This script trains machine learning models using preprocessed data.
# It includes model definition, hyperparameter tuning, and training.
# Executed during the CI/CD pipeline to update models with new data or retrain periodically.

from sklearn.svm import SVC
from data.data_loading import load_datasets
from models.model_loading import load_models, save_models
from models.model_training import train_model

[X_train, X_validation, Y_train, Y_validation] = load_datasets(["X_train", "X_validation", "Y_train", "Y_validation"])

print("Loading previous model")
model_name = "finalised_model"
model = load_models(model_name)



if model == []:
    print("Creating new model")
    model = SVC(gamma='auto')

# model.fit(X_train, Y_train)
train_model(model, X_train, Y_train)
save_models(model, model_name)

