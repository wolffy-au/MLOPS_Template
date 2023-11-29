# Example data_loading.py

import pandas as pd
import joblib as jl
import os
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

# Data Loading Functions: This file may contain functions for loading data from various sources, such as CSV files, databases, APIs, or other formats. The goal is to abstract away the details of data loading so that it can be easily reused throughout the project.
def load_csv_data(file_path):
    return pd.read_csv(file_path)

def load_database_data(connection_string, query):
    # Code to connect to a database and fetch data
    pass

def load_api_data(api_endpoint):
    # Code to make API requests and fetch data
    pass

# Data Exploration Functions: In some cases, you might include functions for exploring the loaded data, such as summary statistics, distribution plots, or other exploratory data analysis (EDA) tasks.
def explore_dataset(dataset):
    # Code for data exploration tasks
    print(dataset.shape)
    print()
    print(dataset.head(20))
    print()
    print(dataset.describe())
    print()
    print(dataset.groupby('class').size())
    print()

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    # histograms
    dataset.hist()
    plt.show()

    # scatter plot matrix
    scatter_matrix(dataset)
    plt.show()

SAVE_DATASET_PATH = os.path.normpath("data/processed/")

def save_datasets(datasets, filenames, save_path=SAVE_DATASET_PATH):
    save_path = os.path.normpath(save_path)
    if isinstance(filenames, str):
        filenames = [filenames]
    for (dataset, filename) in zip(datasets, filenames):
        dataset_path = os.path.join(save_path, f"{filename}.dataset")
        jl.dump(dataset, dataset_path)

def load_datasets(filenames, save_path=SAVE_DATASET_PATH):
    save_path = os.path.normpath(save_path)
    if isinstance(filenames, str):
        filenames = [filenames]
    datasets = []
    for filename in filenames:
        dataset_path = os.path.join(save_path, f"{filename}.dataset")
        if os.path.exists(dataset_path):
            datasets.append(jl.load(dataset_path))

    return datasets

def clear_datasets(filenames, save_path=SAVE_DATASET_PATH):
    save_path = os.path.normpath(save_path)
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        dataset_path = os.path.join(save_path, f"{filename}.dataset")
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
