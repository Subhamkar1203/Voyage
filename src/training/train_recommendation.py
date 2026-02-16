"""
Training pipeline for Route Recommendation model.
Builds collaborative + content-based models from flights.csv.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
import json

from src.data.data_loader import load_flights
from src.data.preprocessing import preprocess_flights
from src.features.feature_engineering import get_user_route_matrix, get_route_profiles
from src.models.route_recommendation import build_models, evaluate


def run(data_path=None):
    """Complete training pipeline for route recommendation."""
    print("=" * 60)
    print("  Route Recommendation — Training Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_flights(data_path)
    print(f"\n[DATA] Loaded {len(df):,} rows")

    # 2. Preprocess
    df = preprocess_flights(df)

    # 3. Build matrices
    user_route_matrix, _ = get_user_route_matrix(df)
    route_profiles = get_route_profiles(df)
    print(f"[FEAT] User-route matrix: {user_route_matrix.shape}")
    print(f"   Route profiles:    {len(route_profiles)} routes")

    # 4. Train
    mlflow.set_experiment("route_recommendation")
    with mlflow.start_run(run_name="recommendation_build"):
        mlflow.log_param("total_users", user_route_matrix.shape[0])
        mlflow.log_param("total_routes", user_route_matrix.shape[1])
        mlflow.log_param("dataset_rows", len(df))

        knn, route_sim_df = build_models(df, user_route_matrix, route_profiles)

        # 5. Evaluate
        print("\n[EVAL] Evaluating recommendation quality...")
        hit_rate, coverage = evaluate(user_route_matrix, route_sim_df, knn, k=5)
        print(f"   Hit Rate @5:  {hit_rate:.4f}")
        print(f"   Coverage @5:  {coverage:.4f}")

        mlflow.log_metric("hit_rate_at_5", hit_rate)
        mlflow.log_metric("coverage_at_5", coverage)

    print("\n[DONE] Route Recommendation training complete!")
    print("   Models saved to outputs/models/")
    return {"hit_rate": hit_rate, "coverage": coverage}


if __name__ == "__main__":
    run()
