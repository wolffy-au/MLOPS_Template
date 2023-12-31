# 1. Data Processing and Preparation Script (data_processing.py):
# This script handles tasks such as data cleaning, feature engineering, and data preprocessing.
# It is typically executed as a part of the CI/CD pipeline to ensure consistent data processing.
# May be triggered whenever new data is available or as a scheduled task.

from libmlops.data.data_loading import load_csv_data, explore_dataset, save_datasets
from libmlops.data.data_preprocessing import get_xy, split_train_test
from libmlops.features.feature_engineering import (
    label_encode_categorical_features,
    ordinal_encode_categorical_features,
    one_hot_encode_categorical_features,
)
from libmlops.utils.classifier_evaluation import (
    algorithm_evaluation,
    features_evaluation,
    compare_algorithms,
)
from libmlops.utils.features_evaluation import keep_features


def run_data_processing():
    # Load dataset
    file_path = "uc02featureencode/data/external/breast-cancer.csv"
    names = [
        "age",
        "menopause",
        "tumor-size",
        "inv-nodes",
        "node-caps",
        "deg-malig",
        "breast",
        "breast-quad",
        "irradiat",
        "class",
    ]
    dataset = load_csv_data(file_path, names)
    # explore_dataset(dataset)
    print("Sample 3:", dataset[:3], dataset.shape)

    # format all fields
    # Define the mapping of columns to desired data types
    column_types = {
        "age": "str",
        "menopause": "str",
        "tumor-size": "str",
        "inv-nodes": "str",
        "node-caps": "str",
        "deg-malig": "str",
        "breast": "str",
        "breast-quad": "str",
        "irradiat": "str",
        "class": "str",
    }

    dataset = clean_data(dataset, column_types)
    print("Sample 3:", dataset[:3], dataset.shape)
    dataset = encode_categorical_features(dataset)
    dataset["class"] = dataset.pop("class")
    dataset["class"] = dataset["class"].astype(bool)
    print("Dataset columns and shape:", dataset.columns, dataset.shape)
    print("Class:", dataset["class"])

    X, Y = get_xy(dataset)

    # features = identify_features(X, Y)
    # print("Features:", features)
    # dataset = keep_features(dataset, features, keep_y=True)
    # print("X,Y,shape:", X, Y, dataset.shape)

    results, names = algorithm_evaluation(X, Y, verbose=True)
    # compare_algorithms(results, names)

    X_train, X_validation, Y_train, Y_validation = split_train_test(
        dataset, random_state=1
    )
    print("X_train:", X_train.shape)
    print("X_validation:", X_validation.shape)
    print("Y_train:", Y_train.shape)
    print("Y_validation:", Y_validation.shape)

    save_datasets(
        [X_train, X_validation, Y_train, Y_validation],
        ["X_train", "X_validation", "Y_train", "Y_validation"],
        "uc02featureencode/data/processed/",
    )
    pass


def load_data(file_path, names=[]):
    if names == []:
        return load_csv_data(file_path)
    else:
        return load_csv_data(file_path, names=names)


# Data Cleaning: Functions for cleaning the data, handling missing values, and addressing any inconsistencies in the dataset.
def handle_missing_values(data):
    # Code to fill or drop missing values
    pass


def clean_data(dataset, column_types):
    # Code for general data cleaning tasks
    # Convert specific columns to the desired data types
    dataset = dataset.dropna()
    dataset = dataset.astype(column_types)
    return dataset


def encode_categorical_features(dataset):
    # Code for encoding categorical features
    # dataset = label_encode_categorical_features(dataset, dataset.columns)
    dataset = label_encode_categorical_features(dataset, [dataset.columns[-1]])
    # dataset = ordinal_encode_categorical_features(dataset, dataset.columns[-1])
    dataset = one_hot_encode_categorical_features(dataset, dataset.columns[:-1])
    return dataset


def identify_features(X, Y):
    return features_evaluation(X, Y, verbose=True)


if __name__ == "__main__":
    run_data_processing()
