# Voyage Analytics — Corporate Travel Intelligence Platform

A comprehensive ML-powered platform for analyzing Brazilian corporate flight data. Uses **only `flights.csv`** to deliver price prediction, flight class classification, and route recommendations.

---

## Architecture

voyage-analytics-platform/
├── api/                     # Flask REST API
├── src/                     # Core ML source code
├── streamlit_app/           # Streamlit ML dashboard
├── docker/                  # Dockerfiles
├── kubernetes/              # K8s manifests
├── mlflow/                  # MLflow tracking config
├── configs/                 # YAML configuration
├── scripts/                 # Shell scripts
├── tests/                   # Unit tests
├── data/raw/                # flights.csv dataset
├── outputs/models/          # Trained model artifacts
├── docker-compose.yaml
├── Makefile
└── requirements.txt

---

## ML Models

| # | Model | Task | Target |
|---|-------|------|--------|
| 1 | Flight Price Regression | Predict ticket price | `price` |
| 2 | Flight Class Classification | Classify flight type | `flightType` |
| 3 | Route Recommendation | Recommend routes | user→route |

---

## Quick Start

### Install Dependencies
pip install -r requirements.txt

### Train Models
python -m src.training.train_flight_price  
python -m src.training.train_flight_classifier  
python -m src.training.train_recommendation  

### Run API
python -m api.app  

### Run Streamlit
streamlit run streamlit_app/app.py  

---

## Docker

docker-compose up --build -d

---

## Kubernetes

kubectl apply -f kubernetes/

---

## Tech Stack

- ML: scikit-learn, XGBoost, LightGBM  
- API: Flask, Gunicorn  
- Dashboard: Streamlit  
- Tracking: MLflow  
- Containerization: Docker  
- Orchestration: Kubernetes  

---

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
