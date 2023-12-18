# Example feature_engineering.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder

# Feature Engineering Functions: This file may include functions for creating new features from existing ones. Feature engineering involves transforming or combining features to improve the performance of machine learning models.
def add_interaction_feature(df, feature1, feature2):
    df[f'{feature1}_{feature2}_interaction'] = df[feature1] * df[feature2]

#  Handling Date and Time Features: If your dataset contains date and time features, you might include functions for extracting relevant information such as day of the week, month, or year.
def extract_date_features(df, date_column):
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['month'] = df[date_column].dt.month

# Scaling and Normalization: Functions for scaling numerical features to a standard range or normalizing them to have a mean of 0 and a standard deviation of 1.
def scale_numerical_features(df, numerical_features):
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Scaling and Normalization: Functions for scaling numerical features to a standard range or normalizing them to have a minimum of 0 and a maximum of 1.
def normalise_numerical_features(df, numerical_features):
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Handling Categorical Features: If your dataset contains categorical features, you might include functions for one-hot encoding or label encoding these features.
def one_hot_encode_categorical_features(df, categorical_features):
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    df.pop(df.columns[0])
    return df

# Handling Categorical Features: If your dataset contains categorical features, you might include functions for one-hot encoding or label encoding these features.
def label_encode_categorical_features(df, categorical_features):
    for feature in categorical_features:
        df[feature], _ = pd.factorize(df[feature])
    return df

# Handling Categorical Features: If your dataset contains categorical features, you might include functions for one-hot encoding or label encoding these features.
def ordinal_encode_categorical_features(df, ordinal_features):
    encoder = OrdinalEncoder(categories='auto')
    df[ordinal_features] = encoder.fit_transform(df[ordinal_features])
    return df