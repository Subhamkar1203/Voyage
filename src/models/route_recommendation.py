"""
Route Recommendation Model
Recommends flight routes to users based on their flight history.
Hybrid: Collaborative Filtering (KNN) + Content-Based (route similarity).
Uses only flights.csv.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs', 'models')


def build_models(df, user_route_matrix, route_profiles):
    """
    Build collaborative filtering (KNN) and content-based similarity models.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Content-Based: Route similarity ──
    numeric_cols = ['avg_price', 'avg_distance', 'avg_time', 'total_flights',
                    'unique_users', 'pct_economic', 'pct_premium', 'pct_firstClass']
    scaler = StandardScaler()
    route_features_scaled = scaler.fit_transform(route_profiles[numeric_cols])

    route_sim = cosine_similarity(route_features_scaled)
    route_sim_df = pd.DataFrame(route_sim, index=route_profiles['route'], columns=route_profiles['route'])

    # ── Collaborative Filtering: User-based KNN ──
    n_neighbors = min(10, len(user_route_matrix) - 1)
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    knn.fit(user_route_matrix.values)

    # Save all artifacts
    joblib.dump(user_route_matrix, os.path.join(MODEL_DIR, 'user_route_matrix.pkl'))
    joblib.dump(route_sim_df, os.path.join(MODEL_DIR, 'route_similarity.pkl'))
    joblib.dump(knn, os.path.join(MODEL_DIR, 'recommendation_knn.pkl'))
    joblib.dump(route_profiles, os.path.join(MODEL_DIR, 'route_profiles.pkl'))

    print(f"[OK] Recommendation models saved")
    print(f"   User-route matrix: {user_route_matrix.shape}")
    print(f"   Route similarity:  {route_sim_df.shape}")
    print(f"   KNN neighbors:     {n_neighbors}")

    return knn, route_sim_df


def hybrid_recommend(user_code, user_route_matrix, route_sim_df, knn_model, top_n=5):
    """
    Hybrid recommendation: collaborative + content-based.
    Returns list of (route, score) tuples.
    """
    recommendations = defaultdict(float)

    if user_code in user_route_matrix.index:
        user_idx = user_route_matrix.index.get_loc(user_code)
        user_vector = user_route_matrix.iloc[user_idx].values.reshape(1, -1)
        user_visited = set(user_route_matrix.columns[user_route_matrix.iloc[user_idx] > 0])

        # Collaborative: find similar users, aggregate their routes
        distances, indices = knn_model.kneighbors(user_vector)
        similar_users = user_route_matrix.iloc[indices[0]]

        for route in user_route_matrix.columns:
            if route not in user_visited:
                neighbor_scores = similar_users[route].values
                weights = 1 / (distances[0] + 1e-6)
                weighted_score = np.average(neighbor_scores, weights=weights)
                recommendations[route] += weighted_score * 0.6

        # Content-based: find routes similar to ones the user already flies
        for visited_route in user_visited:
            if visited_route in route_sim_df.index:
                sims = route_sim_df[visited_route].sort_values(ascending=False)
                for sim_route, sim_score in sims.items():
                    if sim_route not in user_visited and sim_route != visited_route:
                        recommendations[sim_route] += sim_score * 0.4
    else:
        # Cold-start: recommend most popular routes
        popular = user_route_matrix.sum(axis=0).sort_values(ascending=False)
        for route, score in popular.head(top_n).items():
            recommendations[route] = float(score)

    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_recs


def evaluate(user_route_matrix, route_sim_df, knn_model, k=5):
    """Evaluate recommendation quality: hit rate and catalog coverage."""
    hits = 0
    total = 0
    all_recommended = set()

    for user_code in user_route_matrix.index:
        user_routes = set(user_route_matrix.columns[user_route_matrix.loc[user_code] > 0])
        if len(user_routes) < 2:
            continue

        held_out = list(user_routes)[-1]
        temp = user_route_matrix.copy()
        temp.loc[user_code, held_out] = 0

        recs = hybrid_recommend(user_code, temp, route_sim_df, knn_model, top_n=k)
        rec_routes = {r[0] for r in recs}
        all_recommended.update(rec_routes)

        if held_out in rec_routes:
            hits += 1
        total += 1

    hit_rate = hits / total if total > 0 else 0
    coverage = len(all_recommended) / len(user_route_matrix.columns) if len(user_route_matrix.columns) > 0 else 0
    return hit_rate, coverage
