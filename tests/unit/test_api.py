import pytest
from fastapi.testclient import TestClient

from src.inference.api import app

# Create test client
client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "feature_engine_loaded" in data
    assert data["status"] == "healthy"


def test_predict_endpoint_validation():
    """Test prediction endpoint input validation"""
    # Test empty text
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

    # Test missing text field
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_endpoint_with_valid_input():
    """Test prediction endpoint with valid input"""
    # This test will fail if model is not loaded, but that's expected in testing
    sample_text = "Patient presents with chest pain and shortness of breath."
    response = client.post("/predict", json={"text": sample_text})

    # Should either work (200) or fail due to model not being loaded (503)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "top_predictions" in data
