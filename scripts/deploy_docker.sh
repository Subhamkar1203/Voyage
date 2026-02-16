#!/usr/bin/env bash
# Deploy with Docker Compose
set -e
cd "$(dirname "$0")/.."

echo "Building and starting all containers..."
docker-compose up --build -d

echo ""
echo "Services:"
echo "  API:       http://localhost:5000"
echo "  Streamlit: http://localhost:8501"
echo "  MLflow:    http://localhost:5001"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop:      docker-compose down"
