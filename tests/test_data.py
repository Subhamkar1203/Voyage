"""Tests for data loading and preprocessing."""
import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_loader import load_flights
from src.data.preprocessing import preprocess_flights, encode_categoricals


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'flights.csv')


@pytest.fixture
def flights_df():
    if not os.path.exists(DATA_PATH):
        pytest.skip("flights.csv not found in data/raw/")
    return load_flights(DATA_PATH)


def test_load_flights(flights_df):
    assert isinstance(flights_df, pd.DataFrame)
    assert len(flights_df) > 0
    required_cols = ['travelCode', 'userCode', 'from', 'to', 'flightType', 'price', 'time', 'distance', 'agency']
    for col in required_cols:
        assert col in flights_df.columns, f"Missing column: {col}"


def test_preprocess_flights(flights_df):
    df = preprocess_flights(flights_df)
    assert 'route' in df.columns
    assert 'price_per_km' in df.columns
    if 'date' in flights_df.columns:
        assert 'month' in df.columns
        assert 'day_of_week' in df.columns


def test_encode_categoricals(flights_df):
    df = preprocess_flights(flights_df)
    df_encoded, encoders = encode_categoricals(df, fit=True, save_dir=None)
    assert 'from_encoded' in df_encoded.columns
    assert 'to_encoded' in df_encoded.columns
    assert 'agency_encoded' in df_encoded.columns
    assert len(encoders) > 0
