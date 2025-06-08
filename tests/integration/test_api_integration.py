import pytest
import requests
import time
import json

# Base URL for the API (assuming it's running on localhost:8000)
BASE_URL = "http://localhost:8000"

def wait_for_api(max_retries=30, delay=2):
    """Wait for the API to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    return False

class TestAPIIntegration:
    """Integration tests for the API"""
    
    @classmethod
    def setup_class(cls):
        """Setup for the test class"""
        # Wait for API to be ready
        if not wait_for_api():
            pytest.skip("API is not available")
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        # In integration tests, we expect models to be loaded
        assert data["model_loaded"] == True
        assert data["feature_engine_loaded"] == True
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "Medical Document Classifier API" in data["message"]
    
    def test_prediction_valid_input(self):
        """Test prediction with valid medical text"""
        test_cases = [
            "Patient presents with chest pain and shortness of breath. ECG shows normal sinus rhythm.",
            "Surgical procedure performed under general anesthesia. No complications noted.",
            "Patient has history of diabetes mellitus type 2. Blood glucose levels elevated.",
            "Radiological findings show no acute abnormalities. Patient stable.",
            "Consultation requested for cardiology evaluation. Patient has palpitations."
        ]
        
        for text in test_cases:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"text": text}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "top_predictions" in data
            
            # Validate confidence is between 0 and 1
            assert 0 <= data["confidence"] <= 1
            
            # Validate top_predictions structure
            assert isinstance(data["top_predictions"], list)
            assert len(data["top_predictions"]) <= 5
            
            for pred in data["top_predictions"]:
                assert "class" in pred
                assert "confidence" in pred
                assert 0 <= pred["confidence"] <= 1
    
    def test_prediction_invalid_input(self):
        """Test prediction with invalid input"""
        # Empty text
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": ""}
        )
        assert response.status_code == 422
        
        # Missing text field
        response = requests.post(
            f"{BASE_URL}/predict",
            json={}
        )
        assert response.status_code == 422
        
        # Whitespace only
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": "   "}
        )
        assert response.status_code == 422
    
    def test_api_docs(self):
        """Test that API documentation is available"""
        response = requests.get(f"{BASE_URL}/docs")
        assert response.status_code == 200
        
        # Check that it's serving HTML (OpenAPI docs)
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_spec(self):
        """Test that OpenAPI specification is available"""
        response = requests.get(f"{BASE_URL}/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data 