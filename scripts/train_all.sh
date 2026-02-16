#!/usr/bin/env bash
# ═══════════════════════════════════════════════
# Train all models
# ═══════════════════════════════════════════════
set -e
echo "══════════════════════════════════════════"
echo "  Voyage Analytics — Train All Models"
echo "══════════════════════════════════════════"

cd "$(dirname "$0")/.."

echo ""
echo "▶ Step 1/3: Flight Price Regression"
python -m src.training.train_flight_price

echo ""
echo "▶ Step 2/3: Flight Class Classification"
python -m src.training.train_flight_classifier

echo ""
echo "▶ Step 3/3: Route Recommendation"
python -m src.training.train_recommendation

echo ""
echo "✅ All models trained successfully!"
echo "   Models saved to: outputs/models/"
