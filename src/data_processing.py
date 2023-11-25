# 1. Data Processing and Preparation Script (data_processing.py):
# This script handles tasks such as data cleaning, feature engineering, and data preprocessing.
# It is typically executed as a part of the CI/CD pipeline to ensure consistent data processing.
# May be triggered whenever new data is available or as a scheduled task.

from data.data_loading import load_csv_data, explore_dataset
from data.data_preprocessing import split_train_test

# Load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
file_path = "data/external/iris.csv"
dataset = load_csv_data(file_path)

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset.columns = names

explore_dataset(dataset)

X_train, X_validation, Y_train, Y_validation = split_train_test(dataset)

