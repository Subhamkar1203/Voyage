"""
Feature Engineering — Build feature matrices for regression, classification, and recommendation.
All features derived from flights.csv only.
"""
import numpy as np
import pandas as pd


def get_regression_features(df):
    """
    Build feature matrix for flight price prediction.
    Target: price
    Features: encoded categoricals + distance + time + date features
    """
    feature_cols = [
        'from_encoded', 'to_encoded', 'flightType_encoded',
        'agency_encoded', 'distance', 'time',
        'month', 'day_of_week', 'quarter', 'is_weekend'
    ]
    X = df[feature_cols].values
    y = df['price'].values
    return X, y, feature_cols


def get_classification_features(df):
    """
    Build feature matrix for flight class (flightType) prediction.
    Target: flightType_encoded
    Features: encoded origin/dest/agency + distance + time + price + date features
    """
    feature_cols = [
        'from_encoded', 'to_encoded', 'agency_encoded',
        'distance', 'time', 'price',
        'month', 'day_of_week', 'quarter', 'is_weekend'
    ]
    X = df[feature_cols].values
    y = df['flightType_encoded'].values
    return X, y, feature_cols


def get_user_route_matrix(df):
    """
    Build user × route interaction matrix for route recommendation.
    Each cell = number of times a user flew that route.
    """
    route_counts = df.groupby(['userCode', 'route']).size().reset_index(name='flight_count')

    user_route_matrix = route_counts.pivot_table(
        index='userCode', columns='route', values='flight_count', fill_value=0
    )

    return user_route_matrix, route_counts


def get_route_profiles(df):
    """Build feature profiles for each route (for content-based filtering)."""
    route_profiles = df.groupby('route').agg(
        avg_price=('price', 'mean'),
        avg_distance=('distance', 'mean'),
        avg_time=('time', 'mean'),
        total_flights=('travelCode', 'count'),
        unique_users=('userCode', 'nunique'),
        pct_economic=('flightType', lambda x: (x == 'economic').mean()),
        pct_premium=('flightType', lambda x: (x == 'premium').mean()),
        pct_firstClass=('flightType', lambda x: (x == 'firstClass').mean()),
    ).reset_index()

    return route_profiles
