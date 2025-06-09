"""
Production-grade prediction logging and monitoring system.

This module provides:
- Structured prediction logging
- Model drift detection
- Performance metrics tracking
- Data quality monitoring
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
from dataclasses import dataclass, asdict
import hashlib
import os


@dataclass
class PredictionLog:
    """Structured prediction log entry."""
    
    # Request metadata
    timestamp: str
    request_id: str
    model_version: str
    
    # Input data
    input_text: str
    input_hash: str
    input_length: int
    
    # Prediction results
    prediction: str
    confidence: float
    top_predictions: List[Dict[str, Any]]
    
    # Performance metrics
    preprocessing_time_ms: float
    prediction_time_ms: float
    total_time_ms: float
    
    # Feature metrics (for drift detection)
    feature_stats: Dict[str, float]
    
    # Model metadata
    model_loaded_at: str
    feature_engine_version: str
    
    # Infrastructure metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


class PredictionLogger:
    """Production-grade prediction logger with drift detection."""
    
    def __init__(self, model_version: str = "1.0.0"):
        self.model_version = model_version
        self.logger = self._setup_logger()
        
        # Drift detection - store recent predictions for comparison
        self.recent_predictions: List[Dict] = []
        self.max_recent_predictions = 1000
        
        # Performance tracking
        self.performance_stats = {
            "total_predictions": 0,
            "avg_preprocessing_time": 0.0,
            "avg_prediction_time": 0.0,
            "avg_confidence": 0.0,
            "error_count": 0
        }
        
        # Model metadata
        self.model_loaded_at = datetime.now(timezone.utc).isoformat()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging for predictions."""
        logger = logging.getLogger("prediction_logger")
        logger.setLevel(logging.INFO)
        
        # Create handler for prediction logs
        handler = logging.StreamHandler()
        
        # Use JSON formatter for structured logs
        formatter = logging.Formatter(
            '{"level": "%(levelname)s", "logger": "%(name)s", "timestamp": "%(asctime)s", "message": %(message)s}'
        )
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
            
        return logger
    
    def _generate_request_id(self, input_text: str) -> str:
        """Generate unique request ID."""
        timestamp = str(time.time())
        return hashlib.md5(f"{timestamp}{input_text[:50]}".encode()).hexdigest()[:12]
    
    def _hash_input(self, input_text: str) -> str:
        """Create hash of input for deduplication and analysis."""
        return hashlib.sha256(input_text.encode()).hexdigest()[:16]
    
    def _calculate_feature_stats(self, input_text: str, feature_vector: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate feature statistics for drift detection."""
        stats = {
            "text_length": len(input_text),
            "word_count": len(input_text.split()),
            "avg_word_length": np.mean([len(word) for word in input_text.split()]) if input_text.split() else 0,
            "char_diversity": len(set(input_text.lower())) / len(input_text) if input_text else 0,
            "uppercase_ratio": sum(1 for c in input_text if c.isupper()) / len(input_text) if input_text else 0,
            "digit_ratio": sum(1 for c in input_text if c.isdigit()) / len(input_text) if input_text else 0,
            "punctuation_ratio": sum(1 for c in input_text if not c.isalnum() and not c.isspace()) / len(input_text) if input_text else 0
        }
        
        # Add feature vector stats if available
        if feature_vector is not None:
            stats.update({
                "feature_vector_mean": float(np.mean(feature_vector)),
                "feature_vector_std": float(np.std(feature_vector)),
                "feature_vector_min": float(np.min(feature_vector)),
                "feature_vector_max": float(np.max(feature_vector)),
                "feature_vector_zeros": float(np.sum(feature_vector == 0) / len(feature_vector))
            })
            
        return stats
    
    def _get_system_metrics(self) -> Dict[str, Optional[float]]:
        """Get system performance metrics."""
        try:
            import psutil
            process = psutil.Process()
            return {
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_usage_percent": process.cpu_percent()
            }
        except ImportError:
            return {"memory_usage_mb": None, "cpu_usage_percent": None}
    
    def log_prediction(
        self,
        input_text: str,
        prediction: str,
        confidence: float,
        top_predictions: List[Dict[str, Any]],
        preprocessing_time_ms: float,
        prediction_time_ms: float,
        feature_vector: Optional[np.ndarray] = None,
        feature_engine_version: str = "1.0.0"
    ) -> str:
        """Log a prediction with comprehensive metadata."""
        
        # Generate metadata
        request_id = self._generate_request_id(input_text)
        timestamp = datetime.now(timezone.utc).isoformat()
        total_time_ms = preprocessing_time_ms + prediction_time_ms
        
        # Calculate feature statistics
        feature_stats = self._calculate_feature_stats(input_text, feature_vector)
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        # Create prediction log
        log_entry = PredictionLog(
            timestamp=timestamp,
            request_id=request_id,
            model_version=self.model_version,
            input_text=input_text[:500],  # Truncate for storage
            input_hash=self._hash_input(input_text),
            input_length=len(input_text),
            prediction=prediction,
            confidence=confidence,
            top_predictions=top_predictions,
            preprocessing_time_ms=preprocessing_time_ms,
            prediction_time_ms=prediction_time_ms,
            total_time_ms=total_time_ms,
            feature_stats=feature_stats,
            model_loaded_at=self.model_loaded_at,
            feature_engine_version=feature_engine_version,
            memory_usage_mb=system_metrics["memory_usage_mb"],
            cpu_usage_percent=system_metrics["cpu_usage_percent"]
        )
        
        # Log the prediction
        self.logger.info(json.dumps(asdict(log_entry)))
        
        # Update performance stats
        self._update_performance_stats(log_entry)
        
        # Store for drift detection
        self._store_for_drift_detection(log_entry)
        
        return request_id
    
    def _update_performance_stats(self, log_entry: PredictionLog) -> None:
        """Update running performance statistics."""
        self.performance_stats["total_predictions"] += 1
        n = self.performance_stats["total_predictions"]
        
        # Update running averages
        self.performance_stats["avg_preprocessing_time"] = (
            (self.performance_stats["avg_preprocessing_time"] * (n - 1) + log_entry.preprocessing_time_ms) / n
        )
        self.performance_stats["avg_prediction_time"] = (
            (self.performance_stats["avg_prediction_time"] * (n - 1) + log_entry.prediction_time_ms) / n
        )
        self.performance_stats["avg_confidence"] = (
            (self.performance_stats["avg_confidence"] * (n - 1) + log_entry.confidence) / n
        )
    
    def _store_for_drift_detection(self, log_entry: PredictionLog) -> None:
        """Store prediction for drift detection analysis."""
        self.recent_predictions.append({
            "timestamp": log_entry.timestamp,
            "feature_stats": log_entry.feature_stats,
            "confidence": log_entry.confidence,
            "prediction": log_entry.prediction
        })
        
        # Keep only recent predictions
        if len(self.recent_predictions) > self.max_recent_predictions:
            self.recent_predictions = self.recent_predictions[-self.max_recent_predictions:]
    
    def detect_drift(self, window_size: int = 100) -> Dict[str, Any]:
        """Detect potential model drift based on recent predictions."""
        if len(self.recent_predictions) < window_size:
            return {"status": "insufficient_data", "message": f"Need at least {window_size} predictions"}
        
        recent = self.recent_predictions[-window_size:]
        
        # Calculate drift metrics
        drift_metrics = {}
        
        # Confidence drift
        confidences = [p["confidence"] for p in recent]
        drift_metrics["confidence_mean"] = np.mean(confidences)
        drift_metrics["confidence_std"] = np.std(confidences)
        drift_metrics["low_confidence_ratio"] = sum(1 for c in confidences if c < 0.7) / len(confidences)
        
        # Feature drift (simplified)
        feature_means = {}
        for feature in recent[0]["feature_stats"].keys():
            values = [p["feature_stats"][feature] for p in recent]
            feature_means[f"{feature}_mean"] = np.mean(values)
            feature_means[f"{feature}_std"] = np.std(values)
        
        drift_metrics.update(feature_means)
        
        # Prediction distribution drift
        predictions = [p["prediction"] for p in recent]
        unique_predictions = list(set(predictions))
        prediction_distribution = {pred: predictions.count(pred) / len(predictions) for pred in unique_predictions}
        
        # Simple drift detection rules
        drift_alerts = []
        
        if drift_metrics["low_confidence_ratio"] > 0.3:
            drift_alerts.append("HIGH_LOW_CONFIDENCE_RATIO")
        
        if drift_metrics["confidence_std"] > 0.25:
            drift_alerts.append("HIGH_CONFIDENCE_VARIANCE")
        
        return {
            "status": "analyzed",
            "window_size": window_size,
            "drift_metrics": drift_metrics,
            "prediction_distribution": prediction_distribution,
            "drift_alerts": drift_alerts,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        return {
            "model_version": self.model_version,
            "model_loaded_at": self.model_loaded_at,
            "performance_stats": self.performance_stats.copy(),
            "recent_predictions_count": len(self.recent_predictions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def log_error(self, input_text: str, error_message: str, error_type: str) -> None:
        """Log prediction errors."""
        self.performance_stats["error_count"] += 1
        
        error_log = {
            "event_type": "prediction_error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": self._generate_request_id(input_text),
            "model_version": self.model_version,
            "input_hash": self._hash_input(input_text),
            "error_type": error_type,
            "error_message": error_message,
            "total_errors": self.performance_stats["error_count"]
        }
        
        self.logger.error(json.dumps(error_log)) 