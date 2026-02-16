<<<<<<< HEAD
# Voyage Analytics — Corporate Travel Intelligence Platform

A comprehensive ML-powered platform for analyzing Brazilian corporate flight data. Uses **only `flights.csv`** to deliver price prediction, flight class classification, and route recommendations.

---

## Architecture

```
voyage-analytics-platform/
├── api/                     # Flask REST API
│   ├── app.py               # Main application factory
│   ├── config.py            # Configuration classes
│   ├── wsgi.py              # Gunicorn entry point
│   ├── routes/              # API route blueprints
│   │   ├── flight_price.py  # Price prediction endpoint
│   │   ├── flight_class.py  # Class classification endpoint
│   │   └── route_recommend.py # Route recommendation endpoint
│   ├── schemas/             # Request/response schemas
│   └── middleware/          # Error handlers
├── src/                     # Core ML source code
│   ├── data/                # Data loading & preprocessing
│   ├── features/            # Feature engineering
│   ├── models/              # ML models (3 models)
│   ├── training/            # Training pipelines
│   └── utils/               # Helpers & metrics
├── streamlit_app/           # Streamlit ML dashboard
├── docker/                  # Dockerfiles (API, Streamlit, MLflow, Training)
├── kubernetes/              # K8s manifests (namespace, deployments, HPA)
├── mlflow/                  # MLflow tracking config
├── configs/                 # YAML configuration
├── scripts/                 # Shell scripts
├── tests/                   # Unit tests
├── data/raw/                # flights.csv dataset
├── outputs/models/          # Trained model artifacts
├── docker-compose.yaml      # Multi-service orchestration
├── Makefile                 # Common commands
└── requirements.txt         # Python dependencies
```

---

## ML Models

| # | Model | Task | Target | Algorithm |
|---|-------|------|--------|-----------|
| 1 | **Flight Price Regression** | Predict ticket price | `price` | 9 regressors + XGBoost tuning |
| 2 | **Flight Class Classification** | Classify flight type | `flightType` | 7 classifiers |
| 3 | **Route Recommendation** | Recommend routes | user→route | Hybrid KNN + cosine similarity |

### Data: `flights.csv`
| Column | Type | Description |
|--------|------|-------------|
| travelCode | int | Unique travel identifier |
| userCode | int | User identifier |
| from | str | Departure city |
| to | str | Destination city |
| flightType | str | economic / premium / firstClass |
| price | float | Ticket price (BRL) |
| time | float | Flight duration (hours) |
| distance | float | Flight distance (km) |
| agency | str | Airline agency |
| date | str | Flight date |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Dataset
Ensure `flights.csv` is in `data/raw/`.

### 3. Train All Models
```bash
# Train all 3 models with MLflow tracking
python -m src.training.train_flight_price
python -m src.training.train_flight_classifier
python -m src.training.train_recommendation
```

### 4. Start Flask API
```bash
python -m api.app
# API at http://localhost:5000
```

### 5. Start Streamlit Dashboard
```bash
streamlit run streamlit_app/app.py
# Dashboard at http://localhost:8501
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/flight-price/predict` | Predict flight price |
| POST | `/api/v1/flight-class/predict` | Classify flight type |
| GET | `/api/v1/route-recommend/<user_code>` | Get route recommendations |
| GET | `/api/v1/route-recommend/popular` | Popular routes |

### Example: Predict Price
```bash
curl -X POST http://localhost:5000/api/v1/flight-price/predict \
  -H "Content-Type: application/json" \
  -d '{"from":"SaoPaulo","to":"Brasilia","flightType":"economic","agency":"FlyHigh","distance":1015,"time":2.5,"month":6,"day_of_week":2}'
```

---

## Docker Deployment

```bash
# Build and start everything
docker-compose up --build -d

# Services:
#   API:       http://localhost:5000
#   Streamlit: http://localhost:8501
#   MLflow:    http://localhost:5001
```

## Kubernetes Deployment

```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/mlflow/deployment.yaml
kubectl apply -f kubernetes/api/deployment.yaml
kubectl apply -f kubernetes/streamlit/deployment.yaml

# Check status
kubectl get pods -n voyage-analytics
```

Features: HPA auto-scaling (2-8 replicas), liveness/readiness probes, resource limits, ingress routing.

---

## MLflow Tracking

All training runs are logged to MLflow:
- **Experiments:** `flight_price_regression`, `flight_class_classification`, `route_recommendation`
- **Tracking:** Parameters, metrics, and model artifacts
- **UI:** `mlflow ui` at http://localhost:5001

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Tech Stack

- **ML:** scikit-learn, XGBoost, LightGBM
- **API:** Flask, Flask-CORS, Gunicorn
- **Dashboard:** Streamlit, Plotly
- **Tracking:** MLflow
- **Containerization:** Docker, Docker Compose
- **Orchestration:** Kubernetes (HPA, Ingress)
- **Data:** pandas, numpy, scipy
=======
# 🚀 How to Copy This Project to Another GitHub Account

If you want to upload this same project to another GitHub account:

### 1️⃣ Create a new empty repository in your second GitHub account.

### 2️⃣ In your project folder, change the remote URL:

git remote remove origin  
git remote add origin https://github.com/NEW_USERNAME/NEW_REPO_NAME.git  

### 3️⃣ Push to the new repository:

git push -u origin main  

Done 🎉  
Your project is now copied to another GitHub account.
