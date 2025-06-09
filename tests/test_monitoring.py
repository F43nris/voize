"""
Tests for production monitoring components.

Tests cover:
- Prediction logging and drift detection
- Business metrics tracking
- API monitoring endpoints
- Alerting and anomaly detection
"""

import pytest
import time
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.monitoring.prediction_logger import PredictionLogger, PredictionLog
from src.monitoring.business_metrics import BusinessMetricsTracker, RequestMetrics


class TestPredictionLogger:
    """Test the prediction logging and drift detection system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.logger = PredictionLogger(model_version="test-1.0.0")
    
    def test_prediction_logging(self):
        """Test basic prediction logging functionality."""
        request_id = self.logger.log_prediction(
            input_text="Patient has chest pain and shortness of breath",
            prediction="cardiology",
            confidence=0.85,
            top_predictions=[
                {"class": "cardiology", "confidence": 0.85},
                {"class": "emergency", "confidence": 0.12}
            ],
            preprocessing_time_ms=45.2,
            prediction_time_ms=23.8
        )
        
        assert isinstance(request_id, str)
        assert len(request_id) == 12
        assert self.logger.performance_stats["total_predictions"] == 1
        assert len(self.logger.recent_predictions) == 1
    
    def test_feature_statistics_calculation(self):
        """Test feature statistics calculation for drift detection."""
        stats = self.logger._calculate_feature_stats(
            "Patient presents with acute chest pain.",
            feature_vector=np.array([0.1, 0.0, 0.8, 0.3, 0.0])
        )
        
        assert "text_length" in stats
        assert "word_count" in stats
        assert "feature_vector_mean" in stats
        assert "feature_vector_std" in stats
        assert stats["text_length"] == 37
        assert stats["word_count"] == 6
    
    def test_drift_detection_insufficient_data(self):
        """Test drift detection with insufficient data."""
        result = self.logger.detect_drift(window_size=100)
        
        assert result["status"] == "insufficient_data"
        assert "Need at least 100 predictions" in result["message"]
    
    def test_drift_detection_with_data(self):
        """Test drift detection with sufficient data."""
        # Generate sample predictions
        for i in range(150):
            confidence = 0.9 if i < 100 else 0.5  # Simulate confidence drift
            self.logger.log_prediction(
                input_text=f"Test input {i}",
                prediction="test_class",
                confidence=confidence,
                top_predictions=[{"class": "test_class", "confidence": confidence}],
                preprocessing_time_ms=20.0,
                prediction_time_ms=30.0
            )
        
        result = self.logger.detect_drift(window_size=100)
        
        assert result["status"] == "analyzed"
        assert "drift_metrics" in result
        assert "drift_alerts" in result
        assert "HIGH_LOW_CONFIDENCE_RATIO" in result["drift_alerts"]
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Log some predictions
        for i in range(5):
            self.logger.log_prediction(
                input_text=f"Test {i}",
                prediction="test",
                confidence=0.8,
                top_predictions=[],
                preprocessing_time_ms=10.0 + i,
                prediction_time_ms=20.0 + i
            )
        
        summary = self.logger.get_performance_summary()
        
        assert summary["model_version"] == "test-1.0.0"
        assert summary["performance_stats"]["total_predictions"] == 5
        assert summary["recent_predictions_count"] == 5
        assert "timestamp" in summary
    
    def test_error_logging(self):
        """Test error logging functionality."""
        initial_error_count = self.logger.performance_stats["error_count"]
        
        self.logger.log_error(
            input_text="problematic input",
            error_message="Model prediction failed",
            error_type="PREDICTION_ERROR"
        )
        
        assert self.logger.performance_stats["error_count"] == initial_error_count + 1


class TestBusinessMetricsTracker:
    """Test business metrics tracking and analysis."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tracker = BusinessMetricsTracker(window_minutes=60)
    
    def test_request_tracking(self):
        """Test basic request tracking."""
        self.tracker.track_request(
            endpoint="/predict",
            method="POST",
            status_code=200,
            response_time_ms=150.5,
            request_size_bytes=1024,
            response_size_bytes=256
        )
        
        assert len(self.tracker.requests) == 1
        assert len(self.tracker.response_times["/predict"]) == 1
        assert len(self.tracker.errors) == 0
    
    def test_error_tracking(self):
        """Test error request tracking."""
        self.tracker.track_request(
            endpoint="/predict",
            method="POST",
            status_code=500,
            response_time_ms=100.0
        )
        
        assert len(self.tracker.requests) == 1
        assert len(self.tracker.errors) == 1
    
    def test_current_metrics_calculation(self):
        """Test current metrics calculation."""
        # Add some sample requests
        for i in range(10):
            status_code = 500 if i < 2 else 200  # 20% error rate
            response_time = 100 + (i * 10)  # Increasing response times
            
            self.tracker.track_request(
                endpoint="/predict",
                method="POST",
                status_code=status_code,
                response_time_ms=response_time
            )
        
        metrics = self.tracker.get_current_metrics()
        
        assert "metrics" in metrics
        assert metrics["metrics"]["total_requests"] == 10
        assert metrics["metrics"]["total_errors"] == 2
        assert metrics["metrics"]["error_rate_percent"] == 20.0
        assert metrics["metrics"]["avg_response_time_ms"] > 0
    
    def test_endpoint_metrics(self):
        """Test per-endpoint metrics."""
        # Add requests to different endpoints
        for endpoint in ["/predict", "/health", "/monitoring/metrics"]:
            for i in range(5):
                self.tracker.track_request(
                    endpoint=endpoint,
                    method="GET" if endpoint != "/predict" else "POST",
                    status_code=200,
                    response_time_ms=50 + i * 10
                )
        
        predict_metrics = self.tracker.get_endpoint_metrics("/predict")
        
        assert predict_metrics["endpoint"] == "/predict"
        assert predict_metrics["metrics"]["total_requests"] == 5
        assert predict_metrics["metrics"]["error_rate_percent"] == 0.0
        assert predict_metrics["metrics"]["avg_response_time_ms"] > 0
    
    def test_error_summary(self):
        """Test error summary generation."""
        # Add some errors
        error_endpoints = ["/predict", "/predict", "/health"]
        error_codes = [500, 404, 503]
        
        for endpoint, code in zip(error_endpoints, error_codes):
            self.tracker.track_request(
                endpoint=endpoint,
                method="POST",
                status_code=code,
                response_time_ms=100
            )
        
        summary = self.tracker.get_error_summary(hours=24)
        
        assert summary["summary"]["total_errors"] == 3
        assert "/predict" in summary["summary"]["errors_by_endpoint"]
        assert 500 in summary["summary"]["errors_by_status_code"]
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Create high error rate scenario
        for i in range(20):
            status_code = 500 if i < 12 else 200  # 60% error rate
            self.tracker.track_request(
                endpoint="/predict",
                method="POST",
                status_code=status_code,
                response_time_ms=100
            )
        
        anomalies = self.tracker.detect_anomalies()
        
        assert anomalies["status"] == "DEGRADED"
        assert len(anomalies["anomalies"]) > 0
        
        # Check for high error rate anomaly
        error_anomaly = next(
            (a for a in anomalies["anomalies"] if a["type"] == "HIGH_ERROR_RATE"),
            None
        )
        assert error_anomaly is not None
        assert error_anomaly["severity"] == "HIGH"
    
    def test_hourly_trends(self):
        """Test hourly trends calculation."""
        # Simulate requests over time by manually setting hour keys
        with patch.object(self.tracker, 'hourly_stats') as mock_stats:
            mock_stats.__getitem__.return_value = {
                "requests": 100,
                "errors": 5,
                "avg_response_time": 150.0,
                "total_response_time": 15000.0
            }
            mock_stats.__contains__.return_value = True
            
            trends = self.tracker.get_hourly_trends(hours=2)
            
            assert "trends" in trends
            assert trends["period_hours"] == 2


class TestMonitoringAPI:
    """Test the monitoring API endpoints."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app for testing."""
        from fastapi.testclient import TestClient
        from src.inference.api import app
        return TestClient(app)
    
    def test_health_endpoint_with_monitoring(self, mock_app):
        """Test health endpoint includes monitoring data."""
        with patch('src.inference.api.model', Mock()), \
             patch('src.inference.api.feature_engine', Mock()):
            
            response = mock_app.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert "model_loaded" in data
            assert "uptime_seconds" in data
            assert "model_version" in data
    
    def test_monitoring_metrics_endpoint(self, mock_app):
        """Test monitoring metrics endpoint."""
        response = mock_app.get("/monitoring/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "data" in data
        assert "metrics" in data["data"]
    
    def test_monitoring_performance_endpoint(self, mock_app):
        """Test monitoring performance endpoint."""
        response = mock_app.get("/monitoring/performance")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "data" in data
    
    def test_monitoring_drift_endpoint(self, mock_app):
        """Test drift detection endpoint."""
        response = mock_app.get("/monitoring/drift?window_size=50")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "data" in data
    
    def test_monitoring_anomalies_endpoint(self, mock_app):
        """Test anomaly detection endpoint."""
        response = mock_app.get("/monitoring/anomalies")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "data" in data
        assert "anomalies" in data["data"]
    
    def test_monitoring_staleness_endpoint(self, mock_app):
        """Test model staleness monitoring endpoint."""
        response = mock_app.get("/monitoring/staleness")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "data" in data
        assert "recommendation" in data["data"]
    
    def test_enhanced_root_endpoint(self, mock_app):
        """Test enhanced root endpoint with monitoring info."""
        response = mock_app.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "monitoring_features" in data
        assert "health_summary" in data
        assert "monitoring" in data["endpoints"]
        assert len(data["monitoring_features"]) > 0


class TestModelStalenessMonitoring:
    """Test model staleness detection and recommendations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.logger = PredictionLogger(model_version="test-1.0.0")
        self.tracker = BusinessMetricsTracker()
    
    def test_fresh_model_staleness(self):
        """Test staleness check for fresh model."""
        from src.inference.api import check_model_staleness
        
        # Mock recent activity
        with patch.object(self.logger, 'get_performance_summary') as mock_summary, \
             patch.object(self.tracker, 'get_current_metrics') as mock_metrics:
            
            mock_summary.return_value = {
                "performance_stats": {"total_predictions": 100}
            }
            mock_metrics.return_value = {
                "metrics": {"requests_per_minute": 5.0}
            }
            
            with patch('src.inference.api.prediction_logger', self.logger), \
                 patch('src.inference.api.business_metrics', self.tracker):
                
                result = check_model_staleness()
                
                assert len(result["staleness_alerts"]) == 0
                assert "fresh" in result["recommendation"].lower()
    
    def test_stale_model_detection(self):
        """Test detection of stale model."""
        from src.inference.api import check_model_staleness
        
        # Mock old model
        old_time = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        
        with patch.object(self.logger, 'model_loaded_at', old_time), \
             patch.object(self.logger, 'get_performance_summary') as mock_summary, \
             patch.object(self.tracker, 'get_current_metrics') as mock_metrics:
            
            mock_summary.return_value = {
                "performance_stats": {"total_predictions": 10}
            }
            mock_metrics.return_value = {
                "metrics": {"requests_per_minute": 0.05}  # Very low usage
            }
            
            with patch('src.inference.api.prediction_logger', self.logger), \
                 patch('src.inference.api.business_metrics', self.tracker):
                
                result = check_model_staleness()
                
                assert len(result["staleness_alerts"]) >= 1
                assert "retraining" in result["recommendation"].lower()
                
                # Check for specific alert types
                alert_types = [alert["type"] for alert in result["staleness_alerts"]]
                assert "MODEL_AGE_HIGH" in alert_types


if __name__ == "__main__":
    pytest.main([__file__]) 