"""
Route Recommendation API routes.
Recommends flight routes for a given user based on their travel history.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import joblib
from flask import Blueprint, request, jsonify
from src.models.route_recommendation import hybrid_recommend

route_recommend_bp = Blueprint('route_recommend', __name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'models')


def _load_artifacts():
    user_route_matrix = joblib.load(os.path.join(MODEL_DIR, 'user_route_matrix.pkl'))
    route_sim_df = joblib.load(os.path.join(MODEL_DIR, 'route_similarity.pkl'))
    knn_model = joblib.load(os.path.join(MODEL_DIR, 'recommendation_knn.pkl'))
    route_profiles = joblib.load(os.path.join(MODEL_DIR, 'route_profiles.pkl'))
    return user_route_matrix, route_sim_df, knn_model, route_profiles


@route_recommend_bp.route('/<int:user_code>', methods=['GET'])
def recommend(user_code):
    """
    Get route recommendations for a user.
    Query params: top_n (default 5)
    """
    top_n = request.args.get('top_n', 5, type=int)
    top_n = min(max(top_n, 1), 20)

    try:
        user_route_matrix, route_sim_df, knn_model, route_profiles = _load_artifacts()
    except FileNotFoundError:
        return jsonify({"error": "Recommendation model not trained yet. Run training pipeline first."}), 503

    recommendations = hybrid_recommend(user_code, user_route_matrix, route_sim_df, knn_model, top_n=top_n)

    result = []
    for route, score in recommendations:
        entry = {"route": route, "score": round(score, 4)}
        # Add route profile info if available
        profile = route_profiles[route_profiles['route'] == route]
        if len(profile) > 0:
            row = profile.iloc[0]
            entry["avg_price"] = round(float(row.get('avg_price', 0)), 2)
            entry["avg_distance"] = round(float(row.get('avg_distance', 0)), 1)
            entry["total_flights"] = int(row.get('total_flights', 0))
        result.append(entry)

    is_known = user_code in user_route_matrix.index
    return jsonify({
        "user_code": user_code,
        "known_user": is_known,
        "recommendations": result,
        "count": len(result)
    })


@route_recommend_bp.route('/popular', methods=['GET'])
def popular_routes():
    """Get most popular routes overall."""
    top_n = request.args.get('top_n', 10, type=int)
    try:
        user_route_matrix, _, _, route_profiles = _load_artifacts()
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet."}), 503

    popular = user_route_matrix.sum(axis=0).sort_values(ascending=False).head(top_n)
    result = [{"route": route, "flight_count": int(count)} for route, count in popular.items()]
    return jsonify({"popular_routes": result})


@route_recommend_bp.route('/info', methods=['GET'])
def info():
    return jsonify({
        "model": "Route Recommendation",
        "description": "Hybrid collaborative + content-based route recommendation from flights.csv",
        "endpoints": {
            "recommend": "GET /<user_code>?top_n=5",
            "popular": "GET /popular?top_n=10"
        }
    })
