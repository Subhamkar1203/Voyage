"""
Preprocessing — Clean, parse dates, handle missing values for flights.csv.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def preprocess_flights(df):
    """
    Clean and preprocess raw flights DataFrame.
    Returns processed DataFrame with engineered date features.
    """
    df = df.copy()

    # Parse dates
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')

    # Drop rows with null dates (if any)
    df = df.dropna(subset=['date'])

    # Time-based features
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Route feature
    df['route'] = df['from'] + ' -> ' + df['to']

    # Price per km
    df['price_per_km'] = np.where(df['distance'] > 0, df['price'] / df['distance'], 0)

    print(f"[OK] Preprocessing complete: {df.shape[0]:,} rows, {df.shape[1]} cols")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Flight classes: {df['flightType'].value_counts().to_dict()}")

    return df


def encode_categoricals(df, fit=True, encoders=None, save_dir=None):
    """
    Encode categorical columns using LabelEncoder.
    If fit=True, fit new encoders and optionally save them.
    If fit=False, use provided encoders to transform.
    Returns encoded df and dict of encoders.
    """
    df = df.copy()
    cat_cols = ['from', 'to', 'flightType', 'agency']

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        col_encoded = f'{col}_encoded'
        if fit:
            le = LabelEncoder()
            df[col_encoded] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels
            known = set(le.classes_)
            df[col_encoded] = df[col].apply(lambda x: le.transform([x])[0] if x in known else -1)

    if fit and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(encoders, os.path.join(save_dir, 'label_encoders.pkl'))
        print(f"[OK] Encoders saved to {save_dir}")

    return df, encoders
