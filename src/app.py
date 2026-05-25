"""
src/app.py
Flask REST API — loads the saved model and serves live predictions.

Endpoints
---------
POST /predict      → single prediction for hour + day_of_week + month + is_weekend
POST /predict/bulk → array of inputs, returns array of predictions
GET  /health       → health check
GET  /model-info   → metadata about the deployed model
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)

# ── Load artifacts once at startup ──────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model  = joblib.load(os.path.join(BASE, "models", "best_model.pkl"))
scaler = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))

FEATURE_ORDER = [
    "hour_sin", "hour_cos",
    "day_sin",  "day_cos",
    "month_sin","month_cos",
    "is_weekend", "quarter",
    "lag_1h", "lag_24h", "lag_168h", "rolling_24h",
]

def _encode(hour, day_of_week, month, is_weekend, quarter,
            lag_1h=0, lag_24h=0, lag_168h=0, rolling_24h=0):
    """Convert raw inputs into the feature vector the model expects."""
    return [
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day_of_week / 7),
        np.cos(2 * np.pi * day_of_week / 7),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        int(is_weekend),
        int(quarter),
        float(lag_1h),
        float(lag_24h),
        float(lag_168h),
        float(rolling_24h),
    ]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/model-info", methods=["GET"])
def model_info():
    metrics_path = os.path.join(BASE, "outputs", "metrics.json")
    metrics = []
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return jsonify({
        "model_type": type(model).__name__,
        "features": FEATURE_ORDER,
        "metrics": metrics,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Request body (JSON):
    {
        "hour": 14,
        "day_of_week": 2,
        "month": 6,
        "is_weekend": 0,
        "quarter": 2,
        "lag_1h": 45.2,        ← optional, defaults to 0
        "lag_24h": 43.0,       ← optional
        "lag_168h": 42.5,      ← optional
        "rolling_24h": 44.1    ← optional
    }
    """
    body = request.get_json(force=True)
    try:
        features = _encode(
            hour        = body["hour"],
            day_of_week = body["day_of_week"],
            month       = body["month"],
            is_weekend  = body.get("is_weekend", int(body["day_of_week"] >= 5)),
            quarter     = body.get("quarter", (body["month"] - 1) // 3 + 1),
            lag_1h      = body.get("lag_1h", 0),
            lag_24h     = body.get("lag_24h", 0),
            lag_168h    = body.get("lag_168h", 0),
            rolling_24h = body.get("rolling_24h", 0),
        )
        X = np.array([features])
        pred = model.predict(X)[0]
        return jsonify({
            "predicted_energy_kWh": round(float(pred), 3),
            "inputs": body,
            "model": type(model).__name__,
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400


@app.route("/predict/bulk", methods=["POST"])
def predict_bulk():
    """
    Request body: list of objects with same fields as /predict
    Returns: list of predictions in same order.
    """
    items = request.get_json(force=True)
    if not isinstance(items, list):
        return jsonify({"error": "Body must be a JSON array"}), 400
    results = []
    for item in items:
        features = _encode(
            hour        = item["hour"],
            day_of_week = item["day_of_week"],
            month       = item["month"],
            is_weekend  = item.get("is_weekend", int(item["day_of_week"] >= 5)),
            quarter     = item.get("quarter", (item["month"] - 1) // 3 + 1),
            lag_1h      = item.get("lag_1h", 0),
            lag_24h     = item.get("lag_24h", 0),
            lag_168h    = item.get("lag_168h", 0),
            rolling_24h = item.get("rolling_24h", 0),
        )
        pred = model.predict(np.array([features]))[0]
        results.append(round(float(pred), 3))
    return jsonify({"predictions_kWh": results, "count": len(results)})


if __name__ == "__main__":
    print("Starting Energy Forecasting API …")
    app.run(debug=True, host="0.0.0.0", port=5000)
