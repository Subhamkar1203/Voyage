#!/usr/bin/env bash
# Start the Streamlit dashboard
set -e
cd "$(dirname "$0")/.."
echo "Starting Voyage Analytics Streamlit on port 8501..."
streamlit run streamlit_app/app.py --server.port 8501
