#!/usr/bin/env bash
# Start the Flask API server (development)
set -e
cd "$(dirname "$0")/.."
echo "Starting Voyage Analytics API on port 5000..."
python -m api.app
