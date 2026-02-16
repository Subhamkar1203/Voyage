"""Tests for feature engineering."""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import load_flights
from src.data.preprocessing import preprocess_flights, encode_categoricals
from src.features.feature_engineering import (
    get_regression_features, get_classification_features,
    get_user_route_matrix, get_route_profiles
)


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'flights.csv')


@pytest.fixture
def processed_df():
    if not os.path.exists(DATA_PATH):
        pytest.skip("flights.csv not found")
    df = load_flights(DATA_PATH)
    df = preprocess_flights(df)
    df, _ = encode_categoricals(df, fit=True, save_dir=None)
    return df


def test_regression_features(processed_df):
    X, y, cols = get_regression_features(processed_df)
    assert X.shape[0] == len(processed_df)
    assert X.shape[1] == len(cols)
    assert len(y) == len(processed_df)


def test_classification_features(processed_df):
    X, y, cols = get_classification_features(processed_df)
    assert X.shape[0] > 0
    assert X.shape[1] == len(cols)


def test_user_route_matrix(processed_df):
    matrix, _ = get_user_route_matrix(processed_df)
    assert matrix.shape[0] > 0  # users
    assert matrix.shape[1] > 0  # routes


def test_route_profiles(processed_df):
    profiles = get_route_profiles(processed_df)
    assert 'route' in profiles.columns
    assert 'avg_price' in profiles.columns
    assert len(profiles) > 0
