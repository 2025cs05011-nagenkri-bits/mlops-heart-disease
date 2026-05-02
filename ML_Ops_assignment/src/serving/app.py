"""
app.py

Flask service that loads the trained Heart Disease pipeline (preprocessor +
classifier) from disk and exposes a JSON prediction endpoint.

Endpoints:
  GET  /        - service banner
  GET  /form    - browser-friendly HTML form for manual predictions
  GET  /health  - liveness/readiness probe (returns model load status)
  GET  /metrics - Prometheus exposition format (request count, latency,
                  prediction counts by class, model load status)
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

import logging
import os
import time

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.data.preprocess import FEATURE_COLUMNS

MODEL_PATH = os.environ.get("MODEL_PATH", "models/heart_model.pkl")
LABELS = {0: "no_disease", 1: "disease"}

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s",'
           '"logger":"%(name)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("heart-disease-api")

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests served",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions served, labelled by predicted class",
    ["label"],
)
MODEL_LOADED_GAUGE = Gauge(
    "model_loaded",
    "1 if the model artifact is loaded into memory, else 0",
)

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    load_error = None
    logger.info("model loaded from %s", MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    model = None
    model_loaded = False
    load_error = str(exc)
    logger.error("model load failed from %s: %s", MODEL_PATH, exc)

MODEL_LOADED_GAUGE.set(1 if model_loaded else 0)


@app.before_request
def _start_timer():
    request._start_time = time.perf_counter()


@app.after_request
def _record_metrics(response):
    start = getattr(request, "_start_time", None)
    if start is not None:
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.path,
        ).observe(elapsed)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.path,
        http_status=response.status_code,
    ).inc()
    return response


@app.get("/")
def root():
    return jsonify({
        "service": "heart-disease-prediction",
        "status": "ok" if model_loaded else "model_not_loaded",
        "endpoints": ["/form", "/health", "/metrics", "/predict"],
    })


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


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
        logger.exception("inference failed")
        return jsonify({"error": "inference failed", "detail": str(exc)}), 500

    response = [
        {
            "prediction": int(p),
            "label": LABELS[int(p)],
            "disease_probability": round(float(prob), 4),
        }
        for p, prob in zip(preds, probas)
    ]
    for item in response:
        PREDICTION_COUNT.labels(label=item["label"]).inc()
    logger.info(
        "predict batch_size=%d labels=%s",
        len(response),
        [r["label"] for r in response],
    )
    return jsonify(response[0] if not isinstance(data, list) else response), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
