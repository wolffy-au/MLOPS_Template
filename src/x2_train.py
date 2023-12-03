# 2. Model Training Script (train.py):
# This script trains machine learning models using preprocessed data.
# It includes model definition, hyperparameter tuning, and training.
# Executed during the CI/CD pipeline to update models with new data or retrain periodically.

from sklearn.svm import SVC
from data.data_loading import load_datasets
from models.model_loading import load_models, save_models
from models.model_training import train_model
from models.model_evaluation import confusion_matrix_model, plot_confusion_matrix

def run_train():
    # print("Loading training datasets")
    # [X_train, Y_train] = load_datasets(["X_train", "Y_train"])

    # print("Loading previous model")
    # model_name = "finalised_model"
    # [model] = load_models(model_name)

    # if model == []:
    #     print("Does not exist - creating new model")
    #     model = SVC(gamma='auto')
    #     print("Training model")
    #     train_model(model, X_train, Y_train)
    #     print("Saving model")
    #     save_models(model, model_name)
    
    # plot_confusion_matrix(confusion_matrix_model(model, X_train, Y_train), model)
    pass

if __name__ == "__main__":
    run_train()
