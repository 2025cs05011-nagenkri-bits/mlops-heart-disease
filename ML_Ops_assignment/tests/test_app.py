"""
test_app.py

End-to-end tests for the Flask service using its in-process test client.
"""


def test_root_returns_service_banner(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["service"] == "heart-disease-prediction"
    assert body["status"] == "ok"
    assert "/predict" in body["endpoints"]


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["model_loaded"] is True
    assert body["model_path"].endswith("heart_model.pkl")


def test_form_endpoint_returns_html(client):
    resp = client.get("/form")
    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    html = resp.get_data(as_text=True)
    assert "<form" in html.lower()
    assert "predict" in html.lower()


def test_predict_high_risk_returns_disease(client, sample_high_risk):
    resp = client.post("/predict", json=sample_high_risk)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] == 1
    assert body["label"] == "disease"
    assert body["disease_probability"] > 0.8


def test_predict_low_risk_returns_no_disease(client, sample_low_risk):
    resp = client.post("/predict", json=sample_low_risk)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] == 0
    assert body["label"] == "no_disease"
    assert body["disease_probability"] < 0.2


def test_predict_batch_returns_list(client, sample_high_risk, sample_low_risk):
    resp = client.post("/predict", json=[sample_high_risk, sample_low_risk])
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body, list)
    assert len(body) == 2
    assert body[0]["label"] == "disease"
    assert body[1]["label"] == "no_disease"


def test_predict_missing_fields_returns_400(client):
    resp = client.post("/predict", json={"age": 63})
    assert resp.status_code == 400
    body = resp.get_json()
    assert "missing" in body["error"].lower()
    assert "required" in body


def test_predict_non_json_body_returns_400(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400
    body = resp.get_json()
    assert "json" in body["error"].lower()


def test_predict_response_keys(client, sample_high_risk):
    """Response must contain exactly the documented keys."""
    resp = client.post("/predict", json=sample_high_risk)
    body = resp.get_json()
    assert set(body.keys()) == {"prediction", "label", "disease_probability"}


def test_metrics_endpoint_exposes_prometheus_format(client, sample_high_risk):
    client.post("/predict", json=sample_high_risk)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.mimetype.startswith("text/plain")
    body = resp.get_data(as_text=True)
    assert "http_requests_total" in body
    assert "http_request_duration_seconds" in body
    assert "model_predictions_total" in body
    assert "model_loaded" in body
