# 1. Data Processing and Preparation Script (data_processing.py):
# This script handles tasks such as data cleaning, feature engineering, and data preprocessing.
# It is typically executed as a part of the CI/CD pipeline to ensure consistent data processing.
# May be triggered whenever new data is available or as a scheduled task.

from libmlops.data.data_loading import load_csv_data, explore_dataset, save_datasets
from libmlops.data.data_preprocessing import get_xy, split_train_test_xy
from libmlops.utils.classifier_evaluation import (
    algorithm_evaluation,
    features_evaluation,
    compare_algorithms,
)
from libmlops.utils.features_evaluation import keep_features


def run_data_processing():
    # Load dataset
    # url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    file_path = "uc01mltutorial/data/external/iris.csv"
    dataset = load_data(file_path)
    dataset.columns = [
        "sepal-length",
        "sepal-width",
        "petal-length",
        "petal-width",
        "class",
    ]
    explore_dataset(dataset)

    X, Y = get_xy(dataset)

    print(dataset.shape)
    features = features_evaluation(X, Y, verbose=True)
    print(features)
    # 2. Comment this out if you want to test integer indices
    features = [dataset.columns[v] for v in features]
    print(features)
    # 1. Comment this out if you want to disable feature selection
    X = keep_features(dataset, features)
    print(X.shape, Y.shape)

    results, algors = algorithm_evaluation(X, Y, verbose=True)
    # compare_algorithms(results, algors)

    X_train, X_validation, Y_train, Y_validation = split_train_test_xy(
        X, Y, random_state=1
    )
    save_datasets(
        [X_train, X_validation, Y_train, Y_validation],
        ["X_train", "X_validation", "Y_train", "Y_validation"],
        "uc01mltutorial/data/processed/",
    )
    return features


def load_data(file_path, names=[]):
    if names == []:
        return load_csv_data(file_path)
    else:
        return load_csv_data(file_path, names=names)


if __name__ == "__main__":
    run_data_processing()
