"""
Training pipeline for Flight Class Classification model.
Predicts flightType (economic/premium/firstClass) using flights.csv.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
import mlflow.sklearn
import json

from src.data.data_loader import load_flights
from src.data.preprocessing import preprocess_flights, encode_categoricals
from src.features.feature_engineering import get_classification_features
from src.models.flight_class_classification import train_all_classifiers


def run(data_path=None):
    """Complete training pipeline for flight class classification."""
    print("=" * 60)
    print("  Flight Class Classification — Training Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_flights(data_path)
    print(f"\n[DATA] Loaded {len(df):,} rows")

    # 2. Preprocess
    df = preprocess_flights(df)
    df, encoders = encode_categoricals(df, fit=True)

    # 3. Features
    X, y, feature_cols = get_classification_features(df)
    class_names = sorted(df['flightType'].unique().tolist()) if 'flightType' in df.columns else None
    print(f"[FEAT] Features: {len(feature_cols)}  Samples: {len(X):,}")
    print(f"   Classes: {class_names}")

    # 4. Train
    mlflow.set_experiment("flight_class_classification")
    with mlflow.start_run(run_name="classification_sweep"):
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("features", json.dumps(feature_cols))
        mlflow.log_param("classes", json.dumps(class_names) if class_names else "unknown")

        results = train_all_classifiers(X, y, feature_cols, class_names=class_names)

        for name, metrics in results.items():
            for metric_name, metric_val in metrics.items():
                if isinstance(metric_val, (int, float)):
                    mlflow.log_metric(f"{name}_{metric_name}", metric_val)

    print("\n[DONE] Flight Class Classification training complete!")
    print("   Models saved to outputs/models/")
    return results


if __name__ == "__main__":
    run()
