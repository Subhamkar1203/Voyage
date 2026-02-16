"""
Flight Class Classification API routes.
Predicts flightType (economic / premium / firstClass).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import Blueprint, request, jsonify
from src.models.flight_class_classification import predict_flight_class

flight_class_bp = Blueprint('flight_class', __name__)


@flight_class_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict flight class.
    JSON body: { "from": "SaoPaulo", "to": "Brasilia", "agency": "FlyHigh",
                 "price": 450.0, "distance": 1015.0, "time": 2.5,
                 "month": 6, "day_of_week": 2 }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400

    required = ['from', 'to', 'agency', 'price', 'distance', 'time']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        predicted_class, confidence = predict_flight_class(data)
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "input": data
        })
    except FileNotFoundError:
        return jsonify({"error": "Model not trained yet. Run training pipeline first."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@flight_class_bp.route('/info', methods=['GET'])
def info():
    return jsonify({
        "model": "Flight Class Classification",
        "description": "Predicts flight class (economic/premium/firstClass) from route and price data",
        "input_schema": {
            "from": "string — departure city",
            "to": "string — destination city",
            "agency": "string — airline agency name",
            "price": "float — ticket price in BRL",
            "distance": "float — flight distance in km",
            "time": "float — flight duration in hours",
            "month": "int (optional) — month 1-12",
            "day_of_week": "int (optional) — 0=Mon..6=Sun"
        },
        "output": "predicted_class + confidence scores"
    })
