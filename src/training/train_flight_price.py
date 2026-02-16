"""
Training pipeline for Flight Price Regression model.
Loads flights.csv, preprocesses, engineers features, trains models, logs to MLflow.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
import mlflow.sklearn
import json

from src.data.data_loader import load_flights
from src.data.preprocessing import preprocess_flights, encode_categoricals
from src.features.feature_engineering import get_regression_features
from src.models.flight_price_regression import train_all_regressors, hyperparameter_tune


def run(data_path=None):
    """Complete training pipeline for flight price prediction."""
    print("=" * 60)
    print("  Flight Price Regression — Training Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_flights(data_path)
    print(f"\n[DATA] Loaded {len(df):,} rows")

    # 2. Preprocess
    df = preprocess_flights(df)
    df, encoders = encode_categoricals(df, fit=True)

    # 3. Features
    X, y, feature_cols = get_regression_features(df)
    print(f"[FEAT] Features: {len(feature_cols)}  Samples: {len(X):,}")

    # 4. Train
    mlflow.set_experiment("flight_price_regression")
    with mlflow.start_run(run_name="regression_sweep"):
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("features", json.dumps(feature_cols))

        results = train_all_regressors(X, y, feature_cols)

        # Log each model's metrics
        for name, metrics in results.items():
            for metric_name, metric_val in metrics.items():
                mlflow.log_metric(f"{name}_{metric_name}", metric_val)

        # 5. Hyperparameter tuning on best
        print("\n[TUNE] Starting hyperparameter tuning (XGBoost)...")
        best_params, best_score = hyperparameter_tune(X, y)
        mlflow.log_params({f"tuned_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("tuned_best_rmse", best_score)

    print("\n[DONE] Flight Price Regression training complete!")
    print("   Models saved to outputs/models/")
    return results


if __name__ == "__main__":
    run()
