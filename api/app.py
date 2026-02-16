"""
Flask REST API — Voyage Analytics Platform
Serves flight price prediction, flight class classification, and route recommendation.
Uses only flights.csv data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, jsonify
from flask_cors import CORS
from api.routes.flight_price import flight_price_bp
from api.routes.flight_class import flight_class_bp
from api.routes.route_recommend import route_recommend_bp
from api.middleware.error_handler import register_error_handlers


def create_app():
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False

    # CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Blueprints
    app.register_blueprint(flight_price_bp, url_prefix='/api/v1/flight-price')
    app.register_blueprint(flight_class_bp, url_prefix='/api/v1/flight-class')
    app.register_blueprint(route_recommend_bp, url_prefix='/api/v1/route-recommend')

    # Error handlers
    register_error_handlers(app)

    @app.route('/')
    def index():
        return jsonify({
            "service": "Voyage Analytics API",
            "version": "1.0.0",
            "endpoints": {
                "flight_price_predict": "/api/v1/flight-price/predict",
                "flight_class_predict": "/api/v1/flight-class/predict",
                "route_recommend": "/api/v1/route-recommend/<user_code>",
                "health": "/api/v1/health"
            }
        })

    @app.route('/api/v1/health')
    def health():
        return jsonify({"status": "healthy"})

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
