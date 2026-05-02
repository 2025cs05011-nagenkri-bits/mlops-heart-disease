"""
app.py

Flask service that loads the trained Heart Disease pipeline (preprocessor +
classifier) from disk and exposes a JSON prediction endpoint.

Endpoints:
  GET  /        - service banner
  GET  /form    - browser-friendly HTML form for manual predictions
  GET  /health  - liveness/readiness probe (returns model load status)
  POST /predict - accepts a JSON object with the 13 raw features and
                  returns the binary prediction plus disease probability

Run locally:
  python -m src.serving.app
  # then in another terminal:
  curl -X POST http://localhost:5000/predict \
       -H "Content-Type: application/json" \
       -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
            "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,
            "ca":0,"thal":6}'

Environment variables:
  MODEL_PATH - path to the joblib pickle (default: models/heart_model.pkl)
  PORT       - port to bind (default: 5000)
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.data.preprocess import FEATURE_COLUMNS

MODEL_PATH = os.environ.get("MODEL_PATH", "models/heart_model.pkl")
LABELS = {0: "no_disease", 1: "disease"}

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    load_error = None
except Exception as exc:  # noqa: BLE001
    model = None
    model_loaded = False
    load_error = str(exc)


@app.get("/")
def root():
    return jsonify({
        "service": "heart-disease-prediction",
        "status": "ok" if model_loaded else "model_not_loaded",
        "endpoints": ["/form", "/health", "/predict"],
    })


@app.get("/form")
def form():
    return render_template("form.html")


@app.get("/health")
def health():
    payload = {
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
    }
    if not model_loaded:
        payload["error"] = load_error
        return jsonify(payload), 503
    return jsonify(payload), 200


@app.post("/predict")
def predict():
    if not model_loaded:
        return jsonify({"error": "model not loaded", "detail": load_error}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "request body must be JSON"}), 400

    records = data if isinstance(data, list) else [data]

    missing_per_record = [
        [c for c in FEATURE_COLUMNS if c not in record]
        for record in records
    ]
    if any(missing_per_record):
        return jsonify({
            "error": "missing required feature(s)",
            "required": FEATURE_COLUMNS,
            "missing": missing_per_record,
        }), 400

    try:
        df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
        preds = model.predict(df).tolist()
        probas = model.predict_proba(df)[:, 1].tolist()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "inference failed", "detail": str(exc)}), 500

    response = [
        {
            "prediction": int(p),
            "label": LABELS[int(p)],
            "disease_probability": round(float(prob), 4),
        }
        for p, prob in zip(preds, probas)
    ]
    return jsonify(response[0] if not isinstance(data, list) else response), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
