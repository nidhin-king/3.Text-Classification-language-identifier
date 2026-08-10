"""Unit tests for the FastAPI backend."""

from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from src.config import MODEL_DIR

MODEL_FILE = MODEL_DIR / "best_model.joblib"


@pytest.fixture(scope="module")
def client():
    if not MODEL_FILE.exists():
        pytest.skip("Trained model not found; run `python -m src.train` first.")
    from api import app

    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["name"] == "Language Identification API"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["n_languages"] >= 20


def test_languages_endpoint(client):
    response = client.get("/languages")
    assert response.status_code == 200
    body = response.json()
    assert "English" in body["languages"]
    assert body["count"] == len(body["languages"])


def test_predict_endpoint_english(client):
    response = client.post("/predict", json={"text": "Hello, how are you?"})
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "English"
    assert 0 < body["confidence"] <= 1.0
    assert len(body["top_k"]) == 5


def test_predict_endpoint_french(client):
    response = client.post("/predict", json={"text": "Bonjour tout le monde"})
    assert response.status_code == 200
    assert response.json()["language"] == "French"


def test_predict_endpoint_top_k_validation(client):
    response = client.post("/predict", json={"text": "hello", "top_k": 0})
    assert response.status_code == 422


def test_predict_endpoint_empty_text(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


def test_batch_endpoint(client):
    response = client.post(
        "/batch",
        json={"texts": ["Hello world", "Bonjour le monde", "Guten Morgen"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert [r["language"] for r in body["results"]] == ["English", "French", "German"]


def test_docs_endpoint(client):
    response = client.get("/docs")
    assert response.status_code == 200
