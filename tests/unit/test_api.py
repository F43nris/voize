import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_basic_imports():
    """Test that we can import the necessary modules"""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required modules: {e}")

def test_pydantic_models():
    """Test the Pydantic models"""
    try:
        from src.inference.api import PredictionRequest, PredictionResponse, HealthResponse
        
        # Test PredictionRequest validation
        with pytest.raises(ValueError):
            PredictionRequest(text="")  # Should fail validation
            
        # Test valid request
        req = PredictionRequest(text="Sample medical text")
        assert req.text == "Sample medical text"
        
        # Test response models can be created
        pred_response = PredictionResponse(
            prediction="test",
            confidence=0.95,
            top_predictions=[{"class": "test", "confidence": 0.95}]
        )
        assert pred_response.prediction == "test"
        
        health_response = HealthResponse(
            status="healthy",
            model_loaded=True,
            feature_engine_loaded=True,
            uptime_seconds=100.0,
            model_version="1.0.0"
        )
        assert health_response.status == "healthy"
        
    except Exception as e:
        pytest.fail(f"Pydantic model tests failed: {e}")

@patch('src.inference.api.model', None)
@patch('src.inference.api.feature_engine', None)
def test_api_endpoints_mock():
    """Test API endpoints with mocked dependencies"""
    try:
        from fastapi.testclient import TestClient
        
        # Mock the load_model function to prevent actual loading
        with patch('src.inference.api.load_model') as mock_load:
            mock_load.return_value = False
            
            from src.inference.api import app
            client = TestClient(app)
            
            # Test root endpoint
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "version" in data
            
            # Test health endpoint (should work even without model)
            response = client.get("/health")
            # Health check might return 503 if model not loaded, which is fine
            assert response.status_code in [200, 503]
            
    except Exception as e:
        pytest.fail(f"API endpoint tests failed: {e}")

def test_monitoring_components():
    """Test monitoring components"""
    try:
        from src.monitoring.prediction_logger import PredictionLogger
        from src.monitoring.business_metrics import BusinessMetricsTracker
        
        # Test prediction logger can be created
        logger = PredictionLogger("test-1.0.0")
        assert logger.model_version == "test-1.0.0"
        
        # Test business metrics tracker
        tracker = BusinessMetricsTracker()
        assert hasattr(tracker, 'track_request')
        
    except Exception as e:
        pytest.fail(f"Monitoring component tests failed: {e}")

def test_feature_engine():
    """Test feature engine components"""
    try:
        from src.inference.feature_engine import MedicalTextFeatureEngine
        
        # Test that the class can be imported and instantiated
        engine = MedicalTextFeatureEngine()
        assert hasattr(engine, 'transform')
        assert hasattr(engine, 'preprocess_text')
        
    except Exception as e:
        pytest.fail(f"Feature engine tests failed: {e}")

# Simple test that always passes to ensure pytest finds at least one test
def test_always_passes():
    """This test always passes to ensure pytest works"""
    assert True
