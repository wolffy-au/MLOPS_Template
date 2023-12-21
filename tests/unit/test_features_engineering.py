import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from libmlops.features.feature_engineering import (
    add_interaction_feature,
    extract_date_features,
    scale_numerical_features,
    normalise_numerical_features,
    one_hot_encode_categorical_features,
    label_encode_categorical_features,
    ordinal_encode_categorical_features,
)

# Create a sample DataFrame for testing
@pytest.fixture
def sample_dataframe():
    data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'date_column': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01']),
        'numerical_feature': [7, 8, 9],
        'categorical_feature': ['cat1', 'cat2', 'cat1'],
        'ordinal_feature': ['low', 'medium', 'high'],
    }
    return pd.DataFrame(data)

def test_add_interaction_feature(sample_dataframe):
    add_interaction_feature(sample_dataframe, 'feature1', 'feature2')
    assert 'feature1_feature2_interaction' in sample_dataframe.columns
    assert np.array_equal(sample_dataframe['feature1_feature2_interaction'], [4, 10, 18])

def test_extract_date_features(sample_dataframe):
    extract_date_features(sample_dataframe, 'date_column')
    assert 'day_of_week' in sample_dataframe.columns
    assert 'month' in sample_dataframe.columns
    assert 'year' in sample_dataframe.columns

def test_scale_numerical_features(sample_dataframe):
    scale_numerical_features(sample_dataframe, ['numerical_feature'])
    assert 'numerical_feature' in sample_dataframe.columns
    assert sample_dataframe['numerical_feature'].mean() == pytest.approx(0.0)
    assert sample_dataframe['numerical_feature'].std() == pytest.approx(1.224744871391589)

def test_normalise_numerical_features(sample_dataframe):
    normalise_numerical_features(sample_dataframe, ['numerical_feature'])
    assert 'numerical_feature' in sample_dataframe.columns
    assert sample_dataframe['numerical_feature'].min() == pytest.approx(0)
    assert sample_dataframe['numerical_feature'].max() == pytest.approx(1)

def test_one_hot_encode_categorical_features(sample_dataframe):
    df_encoded = one_hot_encode_categorical_features(sample_dataframe, ['categorical_feature'])
    assert 'categorical_feature_cat1' in df_encoded.columns
    assert 'categorical_feature_cat2' in df_encoded.columns

def test_label_encode_categorical_features(sample_dataframe):
    df_encoded = label_encode_categorical_features(sample_dataframe, ['categorical_feature'])
    assert 'categorical_feature' in df_encoded.columns
    assert np.array_equal(df_encoded['categorical_feature'], [0, 1, 0])

def test_ordinal_encode_categorical_features(sample_dataframe):
    df_encoded = ordinal_encode_categorical_features(sample_dataframe, ['ordinal_feature'])
    assert 'ordinal_feature' in df_encoded.columns
    assert np.array_equal(df_encoded['ordinal_feature'], [1.0, 2.0, 0.0])
