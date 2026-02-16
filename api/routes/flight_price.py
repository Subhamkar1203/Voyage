"""
Flight Price Prediction API routes.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import Blueprint, request, jsonify
from src.models.flight_price_regression import predict_price

flight_price_bp = Blueprint('flight_price', __name__)


@flight_price_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict flight price.
    JSON body: { "from": "SaoPaulo", "to": "Brasilia", "flightType": "economic",
                 "agency": "FlyHigh", "distance": 1015.0, "time": 2.5,
                 "month": 6, "day_of_week": 2 }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    required = ['from', 'to', 'flightType', 'agency', 'distance', 'time']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        predicted = predict_price(data)
        return jsonify({
            "predicted_price": round(predicted, 2),
            "currency": "BRL",
            "input": data
        })
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run training pipeline first."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@flight_price_bp.route('/info', methods=['GET'])
def info():
    """Returns model info and expected input schema."""
    return jsonify({
        "model": "Flight Price Regression",
        "description": "Predicts flight ticket prices based on route, class, agency, and timing",
        "input_schema": {
            "from": "string — departure city",
            "to": "string — destination city",
            "flightType": "string — economic | premium | firstClass",
            "agency": "string — airline agency name",
            "distance": "float — flight distance in km",
            "time": "float — flight duration in hours",
            "month": "int (optional) — month 1-12",
            "day_of_week": "int (optional) — 0=Mon..6=Sun"
        },
        "output": "predicted_price (BRL)"
    })
