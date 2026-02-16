#!/usr/bin/env bash
# Deploy to Kubernetes
set -e
cd "$(dirname "$0")/.."

echo "Deploying to Kubernetes..."

kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/mlflow/deployment.yaml
kubectl apply -f kubernetes/api/deployment.yaml
kubectl apply -f kubernetes/streamlit/deployment.yaml

echo ""
echo "✅ Kubernetes deployment complete!"
echo "   kubectl get pods -n voyage-analytics"
