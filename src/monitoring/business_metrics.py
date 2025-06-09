"""
Business metrics tracking for ML API.

Tracks key business and operational metrics:
- Request volume and patterns
- Response time percentiles
- Error rates and patterns
- User behavior analytics
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import statistics
import threading
from dataclasses import dataclass, asdict


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


class BusinessMetricsTracker:
    """Tracks business and operational metrics in real-time."""
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.logger = self._setup_logger()
        
        # Thread-safe collections for metrics
        self._lock = threading.Lock()
        
        # Request tracking
        self.requests: deque = deque(maxlen=10000)  # Keep last 10k requests
        
        # Performance tracking
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Error tracking
        self.errors: deque = deque(maxlen=1000)
        
        # Business metrics aggregates
        self.hourly_stats = defaultdict(lambda: {
            "requests": 0,
            "errors": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0
        })
        
        # Real-time metrics
        self.current_metrics = {
            "requests_per_minute": 0,
            "error_rate_percent": 0.0,
            "avg_response_time_ms": 0.0,
            "p95_response_time_ms": 0.0,
            "p99_response_time_ms": 0.0,
            "total_requests": 0,
            "total_errors": 0,
            "uptime_minutes": 0
        }
        
        self.start_time = datetime.now(timezone.utc)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging for business metrics."""
        logger = logging.getLogger("business_metrics")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"level": "%(levelname)s", "logger": "%(name)s", "timestamp": "%(asctime)s", "message": %(message)s}'
        )
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
            
        return logger
    
    def track_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        request_size_bytes: int = 0,
        response_size_bytes: int = 0,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """Track a single request and update metrics."""
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        metrics = RequestMetrics(
            timestamp=timestamp,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        with self._lock:
            # Store request
            self.requests.append(metrics)
            
            # Track response times by endpoint
            self.response_times[endpoint].append(response_time_ms)
            
            # Track errors
            if status_code >= 400:
                self.errors.append({
                    "timestamp": timestamp,
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "response_time_ms": response_time_ms
                })
            
            # Update hourly aggregates
            hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
            self.hourly_stats[hour_key]["requests"] += 1
            self.hourly_stats[hour_key]["total_response_time"] += response_time_ms
            
            if status_code >= 400:
                self.hourly_stats[hour_key]["errors"] += 1
            
            # Update running averages
            self.hourly_stats[hour_key]["avg_response_time"] = (
                self.hourly_stats[hour_key]["total_response_time"] / 
                self.hourly_stats[hour_key]["requests"]
            )
            
        # Log the request
        self.logger.info(json.dumps({
            "event_type": "request_tracked",
            **asdict(metrics)
        }))
        
        # Update real-time metrics periodically
        if len(self.requests) % 10 == 0:  # Update every 10 requests
            self._update_realtime_metrics()
    
    def _update_realtime_metrics(self) -> None:
        """Update real-time metrics based on recent data."""
        with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(minutes=self.window_minutes)
            
            # Filter recent requests
            recent_requests = [
                r for r in self.requests
                if datetime.fromisoformat(r.timestamp.replace('Z', '+00:00')) > cutoff
            ]
            
            if not recent_requests:
                return
            
            # Calculate metrics
            total_requests = len(recent_requests)
            error_requests = [r for r in recent_requests if r.status_code >= 400]
            total_errors = len(error_requests)
            
            # Response times
            response_times = [r.response_time_ms for r in recent_requests]
            
            self.current_metrics.update({
                "requests_per_minute": total_requests / self.window_minutes,
                "error_rate_percent": (total_errors / total_requests * 100) if total_requests > 0 else 0,
                "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
                "p95_response_time_ms": self._percentile(response_times, 0.95) if response_times else 0,
                "p99_response_time_ms": self._percentile(response_times, 0.99) if response_times else 0,
                "total_requests": len(self.requests),
                "total_errors": len(self.errors),
                "uptime_minutes": (now - self.start_time).total_seconds() / 60
            })
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * (len(sorted_data) - 1))
        return sorted_data[index]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics."""
        self._update_realtime_metrics()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window_minutes": self.window_minutes,
            "metrics": self.current_metrics.copy()
        }
    
    def get_endpoint_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Get metrics for a specific endpoint."""
        with self._lock:
            endpoint_requests = [r for r in self.requests if r.endpoint == endpoint]
            
            if not endpoint_requests:
                return {"endpoint": endpoint, "metrics": {}, "message": "No data available"}
            
            response_times = [r.response_time_ms for r in endpoint_requests]
            errors = [r for r in endpoint_requests if r.status_code >= 400]
            
            return {
                "endpoint": endpoint,
                "metrics": {
                    "total_requests": len(endpoint_requests),
                    "total_errors": len(errors),
                    "error_rate_percent": (len(errors) / len(endpoint_requests) * 100) if endpoint_requests else 0,
                    "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
                    "min_response_time_ms": min(response_times) if response_times else 0,
                    "max_response_time_ms": max(response_times) if response_times else 0,
                    "p50_response_time_ms": self._percentile(response_times, 0.5) if response_times else 0,
                    "p95_response_time_ms": self._percentile(response_times, 0.95) if response_times else 0,
                    "p99_response_time_ms": self._percentile(response_times, 0.99) if response_times else 0,
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours."""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            recent_errors = [
                e for e in self.errors
                if datetime.fromisoformat(e["timestamp"].replace('Z', '+00:00')) > cutoff
            ]
            
            if not recent_errors:
                return {"errors": [], "summary": {}, "message": "No errors in the specified period"}
            
            # Group by status code
            error_by_status = defaultdict(int)
            error_by_endpoint = defaultdict(int)
            
            for error in recent_errors:
                error_by_status[error["status_code"]] += 1
                error_by_endpoint[error["endpoint"]] += 1
            
            return {
                "summary": {
                    "total_errors": len(recent_errors),
                    "time_period_hours": hours,
                    "error_rate_per_hour": len(recent_errors) / hours,
                    "errors_by_status_code": dict(error_by_status),
                    "errors_by_endpoint": dict(error_by_endpoint),
                    "most_common_error": max(error_by_status.items(), key=lambda x: x[1]) if error_by_status else None,
                    "most_problematic_endpoint": max(error_by_endpoint.items(), key=lambda x: x[1]) if error_by_endpoint else None
                },
                "recent_errors": recent_errors[-10:],  # Last 10 errors
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_hourly_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get hourly trends for the last N hours."""
        with self._lock:
            current_hour = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
            
            # Get last N hours of data
            trends = {}
            for i in range(hours):
                hour_dt = datetime.now(timezone.utc) - timedelta(hours=i)
                hour_key = hour_dt.strftime("%Y-%m-%d-%H")
                
                if hour_key in self.hourly_stats:
                    stats = self.hourly_stats[hour_key].copy()
                    stats["error_rate_percent"] = (
                        (stats["errors"] / stats["requests"] * 100) 
                        if stats["requests"] > 0 else 0
                    )
                    trends[hour_key] = stats
                else:
                    trends[hour_key] = {
                        "requests": 0,
                        "errors": 0,
                        "avg_response_time": 0.0,
                        "error_rate_percent": 0.0
                    }
            
            return {
                "trends": trends,
                "period_hours": hours,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """Detect performance and error anomalies."""
        metrics = self.get_current_metrics()["metrics"]
        anomalies = []
        
        # High error rate
        if metrics["error_rate_percent"] > 5.0:
            anomalies.append({
                "type": "HIGH_ERROR_RATE",
                "severity": "HIGH" if metrics["error_rate_percent"] > 10.0 else "MEDIUM",
                "message": f"Error rate is {metrics['error_rate_percent']:.1f}% (threshold: 5.0%)",
                "value": metrics["error_rate_percent"]
            })
        
        # High response time
        if metrics["p95_response_time_ms"] > 5000:
            anomalies.append({
                "type": "HIGH_RESPONSE_TIME",
                "severity": "HIGH" if metrics["p95_response_time_ms"] > 10000 else "MEDIUM",
                "message": f"P95 response time is {metrics['p95_response_time_ms']:.0f}ms (threshold: 5000ms)",
                "value": metrics["p95_response_time_ms"]
            })
        
        # Low request volume (potential downtime)
        if metrics["requests_per_minute"] < 0.1 and metrics["uptime_minutes"] > 10:
            anomalies.append({
                "type": "LOW_REQUEST_VOLUME",
                "severity": "MEDIUM",
                "message": f"Very low request volume: {metrics['requests_per_minute']:.2f} req/min",
                "value": metrics["requests_per_minute"]
            })
        
        return {
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "status": "HEALTHY" if not anomalies else "DEGRADED",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def export_metrics_for_monitoring(self) -> Dict[str, Any]:
        """Export metrics in format suitable for external monitoring systems."""
        current = self.get_current_metrics()
        
        return {
            "timestamp": current["timestamp"],
            "metrics": {
                # Request metrics
                "http_requests_total": current["metrics"]["total_requests"],
                "http_requests_per_minute": current["metrics"]["requests_per_minute"],
                
                # Error metrics
                "http_errors_total": current["metrics"]["total_errors"],
                "http_error_rate_percent": current["metrics"]["error_rate_percent"],
                
                # Response time metrics
                "http_response_time_avg_ms": current["metrics"]["avg_response_time_ms"],
                "http_response_time_p95_ms": current["metrics"]["p95_response_time_ms"],
                "http_response_time_p99_ms": current["metrics"]["p99_response_time_ms"],
                
                # System metrics
                "system_uptime_minutes": current["metrics"]["uptime_minutes"],
            }
        } 