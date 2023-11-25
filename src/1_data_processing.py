# 1. Data Processing and Preparation Script (data_processing.py):
# This script handles tasks such as data cleaning, feature engineering, and data preprocessing.
# It is typically executed as a part of the CI/CD pipeline to ensure consistent data processing.
# May be triggered whenever new data is available or as a scheduled task.

from data.data_loading import load_csv_data, explore_dataset
from data.data_preprocessing import split_train_test
from utils.algorithm_evaluation import algorithm_evaluation, compare_algorithms

# Load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
file_path = "data/external/iris.csv"
dataset = load_csv_data(file_path)

dataset.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# explore_dataset(dataset)

X_train, X_validation, Y_train, Y_validation = split_train_test(dataset, random_state=1)

results, names = algorithm_evaluation(X_train, Y_train)
# compare_algorithms(results, names)
