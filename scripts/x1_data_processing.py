# 1. Data Processing and Preparation Script (data_processing.py):
# This script handles tasks such as data cleaning, feature engineering, and data preprocessing.
# It is typically executed as a part of the CI/CD pipeline to ensure consistent data processing.
# May be triggered whenever new data is available or as a scheduled task.

from libmlops.data.data_loading import load_csv_data, explore_dataset, save_datasets
from libmlops.data.data_preprocessing import get_xy, split_train_test
from libmlops.utils.classifier_evaluation import classifier_evaluation, compare_algorithms
from libmlops.utils.features_evaluation import features_evaluation, keep_features

def run_data_processing():
    # Load dataset
    # url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    # file_path = "data/external/iris.csv"
    # names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    # file_path = "data/external/pima-indians-diabetes.csv"
    # names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    # dataset = load_data(file_path, names)
    # explore_dataset(dataset)

    # features = identify_features(dataset)
    # dataset = keep_features(dataset, features, keep_y=True)
    # print(dataset.shape)

    # X, Y = get_xy(dataset)
    # results, names = algorithm_evaluation(X, Y, verbose=True)
    # compare_algorithms(results, names)

    # X_train, X_validation, Y_train, Y_validation = split_train_test(dataset, random_state=1)
    # save_datasets([X_train, X_validation, Y_train, Y_validation], ["X_train", "X_validation", "Y_train", "Y_validation"])
    pass

def load_data(file_path, names=[]):
    if names == []:
        return load_csv_data(file_path)
    else:
        return load_csv_data(file_path, names=names)

def identify_features(dataset):
    X, Y = get_xy(dataset)
    return features_evaluation(X, Y,verbose=True)

if __name__ == "__main__":
    run_data_processing()
