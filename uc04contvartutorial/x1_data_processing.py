# 1. Data Processing and Preparation Script (data_processing.py):
# This script handles tasks such as data cleaning, feature engineering, and data preprocessing.
# It is typically executed as a part of the CI/CD pipeline to ensure consistent data processing.
# May be triggered whenever new data is available or as a scheduled task.

# import sklearn.datasets
from libmlops.data.data_loading import load_csv_data, explore_dataset, save_datasets
from libmlops.data.data_preprocessing import get_xy, split_train_test_xy
from libmlops.features.feature_selection import convert_indices
from libmlops.utils.regressor_evaluation import (
    algorithm_evaluation,
    features_evaluation,
    compare_algorithms,
)
from libmlops.utils.features_evaluation import keep_features


def run_data_processing():
    # dataset = sklearn.datasets.fetch_california_housing(return_X_y=False, as_frame=True)["frame"]
    # dataset.to_csv("../data/external/california_housing.csv", index=False)

    # Load dataset
    file_path = "uc04contvartutorial/data/external/california_housing.csv"
    dataset = load_csv_data(file_path)
    explore_dataset(dataset, show_ui=False)

    X, Y = get_xy(dataset)
    features = features_evaluation(X, Y, verbose=True)
    features = convert_indices(dataset, features)
    print(features)
    dataset_reduced = keep_features(dataset, features)
    print(dataset_reduced.shape)

    X = dataset_reduced
    results, names = algorithm_evaluation(X, Y.values, verbose=True)
    # compare_algorithms(results, names)

    X_train, X_validation, Y_train, Y_validation = split_train_test_xy(
        X, Y, random_state=1
    )
    save_datasets(
        [X_train, X_validation, Y_train, Y_validation],
        ["X_train", "X_validation", "Y_train", "Y_validation"],
        "uc04contvartutorial/data/processed",
    )

    return features


if __name__ == "__main__":
    run_data_processing()
