# 1. Data Processing and Preparation Script (data_processing.py):
# This script handles tasks such as data cleaning, feature engineering, and data preprocessing.
# It is typically executed as a part of the CI/CD pipeline to ensure consistent data processing.
# May be triggered whenever new data is available or as a scheduled task.

from numpy import set_printoptions
from libmlops.data.data_loading import load_csv_data, explore_dataset, save_datasets
from libmlops.data.data_preprocessing import get_xy, split_train_test_xy
from libmlops.utils.classifier_evaluation import (
    algorithm_evaluation,
    features_evaluation,
    compare_algorithms,
)
from libmlops.features.feature_evaluation import (
    get_feature_importance,
    get_k_best_features,
    get_recursive_feature_elimination,
    get_linear_regression,
    get_decision_tree,
)
from libmlops.utils.features_evaluation import keep_features


def run_data_processing():
    # Load dataset
    file_path = "uc03featureselection/data/external/pima-indians-diabetes.csv"
    names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
    dataset = load_data(file_path, names)
    explore_dataset(dataset)

    X, Y = get_xy(dataset)

    evaluate_features(X, Y)

    features = []
    # 1. Comment this out if you want to disable feature selection
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
        "uc03featureselection/data/processed",
    )
    return features


def evaluate_features(X, Y):
    set_printoptions(precision=2)
    print("get_feature_importance", get_feature_importance(X, Y, verbose=False))
    print("get_k_best_features", get_k_best_features(X, Y, verbose=False))
    print(
        "get_recursive_feature_elimination",
        get_recursive_feature_elimination(X, Y, verbose=False),
    )
    print("get_linear_regression", get_linear_regression(X, Y, verbose=False))
    print("get_decision_tree", get_decision_tree(X, Y, verbose=False))


def load_data(file_path, names=[]):
    if names == []:
        return load_csv_data(file_path)
    else:
        return load_csv_data(file_path, names=names)


if __name__ == "__main__":
    run_data_processing()
