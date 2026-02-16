"""Tests for Flask API endpoints."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index(client):
    resp = client.get('/')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'service' in data
    assert data['service'] == 'Voyage Analytics API'


def test_health(client):
    resp = client.get('/api/v1/health')
    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'healthy'


def test_price_predict_missing_body(client):
    resp = client.post('/api/v1/flight-price/predict', json={})
    # Should return 400 for missing fields
    assert resp.status_code == 400


def test_class_predict_missing_body(client):
    resp = client.post('/api/v1/flight-class/predict', json={})
    assert resp.status_code == 400


def test_price_info(client):
    resp = client.get('/api/v1/flight-price/info')
    assert resp.status_code == 200
    assert 'model' in resp.get_json()


def test_class_info(client):
    resp = client.get('/api/v1/flight-class/info')
    assert resp.status_code == 200


def test_recommend_info(client):
    resp = client.get('/api/v1/route-recommend/info')
    assert resp.status_code == 200
