"""
Data Loader — Loads flights.csv and returns raw DataFrame.
"""
import os
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'raw')


def load_flights(path=None):
    """Load flights dataset."""
    if path is None:
        path = os.path.join(DATA_DIR, 'flights.csv')
    df = pd.read_csv(path)
    print(f"[OK] Loaded flights: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def get_data_summary(df):
    """Print dataset summary."""
    print(f"\nColumns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Nulls:\n{df.isnull().sum()}")
    print(f"Shape: {df.shape}")
    return df.describe()
