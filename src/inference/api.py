import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator

# Import the feature engine class and make it available for pickle
from src.inference.feature_engine import MedicalTextFeatureEngine
from src.monitoring.business_metrics import BusinessMetricsTracker

# Import monitoring components
from src.monitoring.prediction_logger import PredictionLogger

# Add cloud storage support
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("⚠️  Google Cloud Storage client not available - using local files only")

# Monkey patch for pickle deserialization - make the class available 
# in the module namespace that pickle expects
sys.modules["__main__"].MedicalTextFeatureEngine = MedicalTextFeatureEngine

# Configure structured logging with timestamps FIRST
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Also add a handler for Cloud Run structured logging
cloud_handler = logging.StreamHandler()
cloud_handler.setFormatter(
    logging.Formatter(
    '{"timestamp": "%(asctime)s", "severity": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    )
)
logger.addHandler(cloud_handler)

# Add DAG scheduler imports and setup (AFTER logger is defined)
try:
    from pathlib import Path
    
    # Add the data-pipeline directory to Python path
    data_pipeline_path = str(Path(__file__).parent.parent.parent / "data-pipeline")
    if data_pipeline_path not in sys.path:
        sys.path.append(data_pipeline_path)
    
    from ml_dags import setup_ml_scheduler  # type: ignore # noqa
    from dag_scheduler import DAGScheduler  # type: ignore # noqa
    
    # Initialize the scheduler (will be setup during startup)
    dag_scheduler = None
    DAG_SCHEDULER_AVAILABLE = True
    logger.info("🔄 DAG scheduler modules loaded")
    
except ImportError as e:
    logger.warning(f"DAG scheduler not available: {e}")
    dag_scheduler = None
    DAG_SCHEDULER_AVAILABLE = False
    # Create dummy variables to avoid undefined variable warnings
    setup_ml_scheduler = None  # type: ignore
    DAGScheduler = None  # type: ignore

# Log startup information
logger.info("=" * 60)
logger.info("🚀 MEDICAL CLASSIFIER API STARTING UP")
logger.info("=" * 60)
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")

# Log environment information
logger.info("Environment variables:")
for key, value in os.environ.items():
    if not key.startswith(("_", "LS_COLORS")):  # Skip private/noisy vars
        logger.info(f"  {key}={value}")

# Initialize FastAPI app
app = FastAPI(
    title="Medical Document Classifier API",
    description="ML API for classifying medical documents using trained Multinomial Naive Bayes model with comprehensive monitoring",
    version="1.0.0",
)

# Global variables for model and feature engine
model = None
feature_engine = None
startup_start_time = time.time()

# Initialize monitoring components
prediction_logger = PredictionLogger(model_version="1.0.0")
business_metrics = BusinessMetricsTracker(window_minutes=60)

logger.info("📊 Monitoring systems initialized")
logger.info(f"   - Prediction logger: {type(prediction_logger).__name__}")
logger.info(f"   - Business metrics tracker: {type(business_metrics).__name__}")

model_load_status = {
    "started": False,
    "completed": False,
    "error": None,
    "attempts": 0,
    "last_attempt_time": None,
    "load_duration": None,
}


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str
    
    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_predictions: List[dict]
    request_id: Optional[str] = None  # Add request ID for tracking


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_engine_loaded: bool
    uptime_seconds: float
    model_version: str
    performance_summary: Optional[Dict[str, Any]] = None


class MonitoringResponse(BaseModel):
    """Response model for monitoring endpoints."""

    timestamp: str
    data: Dict[str, Any]


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Middleware to track all requests for business metrics."""
    start_time = time.time()

    # Get request info
    user_agent = request.headers.get("user-agent")

    # Handle ASGI scope for getting client IP
    client_ip = None
    if hasattr(request, "client") and request.client:
        client_ip = request.client.host

    try:
        response = await call_next(request)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Estimate response size (basic)
        response_size = getattr(response, "headers", {}).get("content-length", 0)
        if isinstance(response_size, str):
            response_size = int(response_size) if response_size.isdigit() else 0

        # Track the request
        business_metrics.track_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=response.status_code,
            response_time_ms=response_time_ms,
            request_size_bytes=(
                int(request.headers.get("content-length", 0))
                if request.headers.get("content-length", "0").isdigit()
                else 0
            ),
            response_size_bytes=response_size,
            user_agent=user_agent,
            ip_address=client_ip,
        )

        return response

    except Exception as e:
        # Track errors
        response_time_ms = (time.time() - start_time) * 1000
        business_metrics.track_request(
            endpoint=str(request.url.path),
            method=request.method,
            status_code=500,
            response_time_ms=response_time_ms,
            user_agent=user_agent,
            ip_address=client_ip,
        )
        raise e


def download_models_from_cloud():
    """Download model files from cloud storage if they don't exist locally"""
    
    # Check if this is a CI/CD environment (mock models will be used anyway)
    if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
        logger.info("🧪 CI/CD environment detected - will use mock models, skipping cloud download")
        return True
    
    model_path = Path("data/optimized_multinomial_nb.pkl")
    feature_engine_path = Path("data/feature_engine.pkl")
    
    # Check if models already exist locally
    if model_path.exists() and feature_engine_path.exists():
        logger.info("✅ Model files found locally, skipping cloud download")
        return True
    
    # Try to download from cloud storage
    if not GCS_AVAILABLE:
        logger.warning("⚠️  Cloud storage not available and models not found locally")
        return False
    
    try:
        # Configuration from environment variables
        bucket_name = os.getenv("GCS_MODEL_BUCKET", "voize-ml-models")
        project_id = os.getenv("GCP_PROJECT_ID")
        
        if not project_id:
            logger.warning("⚠️  GCP_PROJECT_ID not set - cannot download models")
            return False
        
        logger.info(f"☁️  Downloading models from gs://{bucket_name}/")
        
        # Initialize GCS client
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        
        # Download model files
        models_to_download = [
            ("models/optimized_multinomial_nb.pkl", "data/optimized_multinomial_nb.pkl"),
            ("models/feature_engine.pkl", "data/feature_engine.pkl")
        ]
        
        for cloud_path, local_path in models_to_download:
            logger.info(f"Downloading {cloud_path} → {local_path}")
            blob = bucket.blob(cloud_path)
            
            if blob.exists():
                blob.download_to_filename(local_path)
                logger.info(f"✅ Downloaded {local_path}")
            else:
                logger.error(f"❌ {cloud_path} not found in bucket")
                return False
        
        logger.info("🎉 All models downloaded successfully from cloud storage")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download models from cloud: {str(e)}")
        return False


def load_model():
    """Load the trained model and feature engine with retry logic and detailed logging"""
    global model, feature_engine, model_load_status
    
    load_start_time = time.time()
    model_load_status["started"] = True
    model_load_status["last_attempt_time"] = datetime.now().isoformat()
    model_load_status["attempts"] += 1
    
    logger.info("🔄 Starting model loading process...")
    logger.info(f"Attempt #{model_load_status['attempts']}")
    
    # Check if this is a CI/CD environment - use mock models for testing
    if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
        logger.info("🧪 CI/CD environment detected - creating mock models for testing")
        try:
            # Create mock model and feature engine for testing
            from unittest.mock import MagicMock
            
            # Mock model with basic predict/predict_proba methods
            model = MagicMock()
            model.predict.return_value = [0]  # Mock prediction
            model.predict_proba.return_value = [[0.7, 0.2, 0.1]]  # Mock probabilities
            model.classes_ = ["cardiology", "neurology", "orthopedics"]
            
            # Mock feature engine with transform method
            feature_engine = MagicMock()
            feature_engine.transform.return_value = [[0.1, 0.2, 0.3]]  # Mock features
            feature_engine.label_encoder = MagicMock()
            feature_engine.label_encoder.classes_ = ["cardiology", "neurology", "orthopedics"]
            feature_engine.label_encoder.inverse_transform.return_value = ["cardiology"]
            
            total_load_time = time.time() - load_start_time
            model_load_status["completed"] = True
            model_load_status["load_duration"] = total_load_time
            model_load_status["error"] = None
            
            logger.info("✅ Mock models created successfully for CI/CD testing!")
            logger.info(f"   Total loading time: {total_load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create mock models for CI/CD: {str(e)}")
            model_load_status["error"] = str(e)
            model_load_status["completed"] = False
            return False
    
    # First, try to download models from cloud if needed
    logger.info("☁️  Checking for models in cloud storage...")
    cloud_download_success = download_models_from_cloud()
    if not cloud_download_success:
        logger.warning("⚠️  Cloud model download failed, proceeding with local files")
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        attempt_start_time = time.time()
        try:
            logger.info(
                f"📁 ATTEMPT {attempt + 1}/{max_retries} - File system inspection"
            )
            
            # Log current working directory and file structure for debugging
            current_dir = os.getcwd()
            logger.info(f"Current working directory: {current_dir}")
            
            # List contents of current directory with detailed info
            if os.path.exists("."):
                contents = []
                for item in os.listdir("."):
                    item_path = os.path.join(".", item)
                    if os.path.isdir(item_path):
                        contents.append(f"{item}/ (directory)")
                    else:
                        size = os.path.getsize(item_path)
                        contents.append(f"{item} ({size} bytes)")
                logger.info(f"Root directory contents: {contents}")
            
            # Check if data directory exists with detailed info
            if os.path.exists("data"):
                data_contents = []
                total_size = 0
                for item in os.listdir("data"):
                    item_path = os.path.join("data", item)
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path)
                        total_size += size
                        data_contents.append(f"{item} ({size:,} bytes)")
                    else:
                        data_contents.append(f"{item}/ (directory)")
                
                logger.info(f"Data directory contents: {data_contents}")
                logger.info(
                    f"Total data directory size: {total_size:,} bytes ({total_size/(1024*1024):.2f} MB)"
                )
            else:
                logger.error("❌ Data directory does not exist!")
                raise FileNotFoundError("Data directory not found")
            
            # Get model paths and check them
            model_path = Path("data/optimized_multinomial_nb.pkl")
            feature_engine_path = Path("data/feature_engine.pkl")
            
            logger.info(f"🔍 Checking model files:")
            logger.info(f"  Model path: {model_path.absolute()}")
            logger.info(f"  Feature engine path: {feature_engine_path.absolute()}")
            
            # Detailed file checks
            if not model_path.exists():
                logger.error(f"❌ Model file not found: {model_path.absolute()}")
                raise FileNotFoundError(
                    f"Model file not found: {model_path.absolute()}"
                )
            else:
                model_size = model_path.stat().st_size
                logger.info(
                    f"✅ Model file found: {model_size:,} bytes ({model_size/(1024*1024):.2f} MB)"
                )
                
            if not feature_engine_path.exists():
                logger.error(
                    f"❌ Feature engine file not found: {feature_engine_path.absolute()}"
                )
                raise FileNotFoundError(
                    f"Feature engine file not found: {feature_engine_path.absolute()}"
                )
            else:
                fe_size = feature_engine_path.stat().st_size
                logger.info(
                    f"✅ Feature engine file found: {fe_size:,} bytes ({fe_size/(1024*1024):.2f} MB)"
                )
            
            # Load model with timing
            logger.info("📦 Loading model...")
            model_load_start = time.time()
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            model_load_time = time.time() - model_load_start
            logger.info(
                f"✅ Model loaded successfully in {model_load_time:.2f} seconds"
            )
            logger.info(f"   Model type: {type(model)}")
            if hasattr(model, "classes_"):
                logger.info(f"   Model classes: {len(model.classes_)} classes")
            
            # Load feature engine with timing
            logger.info("🔧 Loading feature engine...")
            fe_load_start = time.time()
            with open(feature_engine_path, "rb") as f:
                feature_engine = pickle.load(f)
            fe_load_time = time.time() - fe_load_start
            logger.info(
                f"✅ Feature engine loaded successfully in {fe_load_time:.2f} seconds"
            )
            logger.info(f"   Feature engine type: {type(feature_engine)}")
            if hasattr(feature_engine, "label_encoder"):
                if hasattr(feature_engine.label_encoder, "classes_"):
                    logger.info(
                        f"   Label encoder classes: {len(feature_engine.label_encoder.classes_)}"
                    )
            
            # Test the models quickly
            logger.info("🧪 Testing models with sample data...")
            test_text = ["Patient presents with chest pain"]
            try:
                test_features = feature_engine.transform(test_text)
                test_pred = model.predict(test_features)
                test_proba = model.predict_proba(test_features)
                logger.info(
                    f"✅ Model test successful - prediction shape: {test_pred.shape}, proba shape: {test_proba.shape}"
                )
            except Exception as test_error:
                logger.error(f"❌ Model test failed: {str(test_error)}")
                raise
            
            total_load_time = time.time() - load_start_time
            model_load_status["completed"] = True
            model_load_status["load_duration"] = total_load_time
            model_load_status["error"] = None
            
            logger.info("🎉 MODEL LOADING COMPLETED SUCCESSFULLY!")
            logger.info(f"   Total loading time: {total_load_time:.2f} seconds")
            logger.info(f"   Model loading time: {model_load_time:.2f} seconds")
            logger.info(f"   Feature engine loading time: {fe_load_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            attempt_time = time.time() - attempt_start_time
            error_msg = f"Attempt {attempt + 1}/{max_retries} failed after {attempt_time:.2f}s: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Exception details:", exc_info=True)
            
            model_load_status["error"] = str(e)
            
            if attempt < max_retries - 1:
                logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                total_time = time.time() - load_start_time
                logger.error(f"💥 ALL ATTEMPTS FAILED after {total_time:.2f} seconds")
                model_load_status["completed"] = False
                return False
    
    return False


@app.on_event("startup")
async def startup_event():
    """Initialize the model and monitoring systems on startup"""
    global model, feature_engine, startup_start_time, dag_scheduler
    
    startup_start_time = time.time()
    
    logger.info("🚀 Starting up Medical Classifier API...")
    
    # Download and load models
    if download_models_from_cloud():
        load_model()
    
    # Initialize DAG scheduler if available
    if DAG_SCHEDULER_AVAILABLE:
        try:
            dag_scheduler = setup_ml_scheduler()
            logger.info("🔄 DAG scheduler initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize DAG scheduler: {e}")
            dag_scheduler = None
    
    startup_duration = time.time() - startup_start_time
    logger.info(f"✅ Startup completed in {startup_duration:.2f} seconds")
    
    if model is not None and feature_engine is not None:
        logger.info("🎯 API ready to serve predictions!")
    else:
        logger.warning("⚠️  API started but model/feature engine not loaded")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with detailed status information and monitoring metrics"""
    check_start_time = time.time()
    logger.info("🏥 Health check requested")
    
    model_loaded = model is not None
    feature_engine_loaded = feature_engine is not None
    
    # Get performance summary from prediction logger
    performance_summary = prediction_logger.get_performance_summary()

    # Get uptime
    uptime = time.time() - startup_start_time

    # Get detailed status
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(uptime, 2),
        "model_loaded": model_loaded,
        "feature_engine_loaded": feature_engine_loaded,
        "model_version": "1.0.0",
        "load_status": dict(model_load_status),  # Copy the status
        "performance_summary": performance_summary,
        "system_info": {},
    }
    
    # Add system info if available
    try:
        import psutil

        memory_info = psutil.virtual_memory()
        health_status["system_info"] = {
            "memory_available_gb": round(memory_info.available / (1024**3), 2),
            "memory_percent": memory_info.percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }
    except:
        pass
    
    logger.info(
        f"Health status: {json.dumps({k: v for k, v in health_status.items() if k != 'performance_summary'}, indent=2)}"
    )
    
    # If models aren't loaded yet, try to load them
    if not model_loaded or not feature_engine_loaded:
        logger.info("🔄 Models not loaded, attempting to load from health check...")
        success = load_model()
        if success:
            model_loaded = model is not None
            feature_engine_loaded = feature_engine is not None
            health_status["model_loaded"] = model_loaded
            health_status["feature_engine_loaded"] = feature_engine_loaded
            health_status["load_status"] = dict(model_load_status)
    
    check_time = time.time() - check_start_time
    health_status["health_check_duration"] = round(check_time, 3)
    
    # Return appropriate status
    if model_loaded and feature_engine_loaded:
        logger.info(f"✅ Health check PASSED in {check_time:.3f}s")
        return HealthResponse(
            status="healthy",
            model_loaded=model_loaded,
            feature_engine_loaded=feature_engine_loaded,
            uptime_seconds=health_status["uptime_seconds"],
            model_version=health_status["model_version"],
            performance_summary=performance_summary,
        )
    else:
        # Still loading or failed to load
        logger.warning(
            f"⚠️  Health check FAILED in {check_time:.3f}s - models not ready"
        )
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "message": "Models are not loaded or failed to load",
                **health_status,
            },
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction on the input text with comprehensive monitoring and logging"""
    
    # Check if model is loaded
    if model is None or feature_engine is None:
        prediction_logger.log_error(
            input_text=request.text,
            error_message="Model not loaded",
            error_type="MODEL_NOT_LOADED",
        )
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs."
        )
    
    # Start timing
    total_start_time = time.time()

    try:
        # Preprocessing phase
        preprocessing_start = time.time()
        text_features = feature_engine.transform([request.text])
        preprocessing_time_ms = (time.time() - preprocessing_start) * 1000
        
        # Prediction phase
        prediction_start = time.time()
        prediction_encoded = model.predict(text_features)[0]
        
        # Decode the prediction back to class name
        prediction = feature_engine.label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]
        
        # Get prediction probabilities for confidence and top predictions
        probabilities = model.predict_proba(text_features)[0]
        prediction_time_ms = (time.time() - prediction_start) * 1000
        
        # Get class names from label encoder
        class_names = feature_engine.label_encoder.classes_
        
        # Create top predictions
        top_indices = np.argsort(probabilities)[-5:][::-1]  # Top 5 predictions
        top_predictions = [
            {"class": class_names[idx], "confidence": float(probabilities[idx])}
            for idx in top_indices
        ]
        
        # Get confidence for the predicted class
        confidence = float(probabilities[prediction_encoded])

        # Log the prediction with comprehensive metadata
        request_id = prediction_logger.log_prediction(
            input_text=request.text,
            prediction=prediction,
            confidence=confidence,
            top_predictions=top_predictions,
            preprocessing_time_ms=preprocessing_time_ms,
            prediction_time_ms=prediction_time_ms,
            feature_vector=(
                text_features.toarray()[0]
                if hasattr(text_features, "toarray")
                else None
            ),
            feature_engine_version="1.0.0",
        )

        total_time_ms = (time.time() - total_start_time) * 1000

        logger.info(
            f"✅ Prediction completed - ID: {request_id}, Time: {total_time_ms:.1f}ms, Confidence: {confidence:.3f}"
        )
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            top_predictions=top_predictions,
            request_id=request_id,
        )
        
    except Exception as e:
        total_time_ms = (time.time() - total_start_time) * 1000
        error_msg = str(e)

        logger.error(f"❌ Prediction error: {error_msg} (Time: {total_time_ms:.1f}ms)")

        # Log the error
        prediction_logger.log_error(
            input_text=request.text,
            error_message=error_msg,
            error_type="PREDICTION_ERROR",
        )

        raise HTTPException(status_code=500, detail=f"Prediction failed: {error_msg}")


# New monitoring endpoints - wrapped in try-except to ensure they register
try:
    @app.get("/monitoring/metrics", response_model=MonitoringResponse)
    async def get_business_metrics():
        """Get current business metrics and performance data"""
        try:
            metrics = business_metrics.get_current_metrics()
            return MonitoringResponse(timestamp=metrics["timestamp"], data=metrics)
        except Exception as e:
            logger.error(f"Error getting business metrics: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


    @app.get("/monitoring/performance", response_model=MonitoringResponse)
    async def get_performance_summary():
        """Get detailed performance summary from prediction logger"""
        try:
            performance = prediction_logger.get_performance_summary()
            return MonitoringResponse(timestamp=performance["timestamp"], data=performance)
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve performance data"
            )


    @app.get("/monitoring/drift", response_model=MonitoringResponse)
    async def get_drift_analysis(window_size: int = 100):
        """Get model drift detection analysis"""
        try:
            drift_analysis = prediction_logger.detect_drift(window_size=window_size)
            return MonitoringResponse(
                timestamp=drift_analysis.get("timestamp", datetime.now().isoformat()),
                data=drift_analysis,
            )
        except Exception as e:
            logger.error(f"Error getting drift analysis: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve drift analysis")


    @app.get("/monitoring/errors", response_model=MonitoringResponse)
    async def get_error_summary(hours: int = 24):
        """Get error summary for the specified time period"""
        try:
            error_summary = business_metrics.get_error_summary(hours=hours)
            return MonitoringResponse(
                timestamp=error_summary["timestamp"], data=error_summary
            )
        except Exception as e:
            logger.error(f"Error getting error summary: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve error summary")


    @app.get("/monitoring/anomalies", response_model=MonitoringResponse)
    async def get_anomaly_detection():
        """Get current anomaly detection results"""
        try:
            anomalies = business_metrics.detect_anomalies()
            return MonitoringResponse(timestamp=anomalies["timestamp"], data=anomalies)
        except Exception as e:
            logger.error(f"Error getting anomaly detection: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve anomaly detection"
            )


    @app.get("/monitoring/trends", response_model=MonitoringResponse)
    async def get_hourly_trends(hours: int = 24):
        """Get hourly trends for requests, errors, and performance"""
        try:
            trends = business_metrics.get_hourly_trends(hours=hours)
            return MonitoringResponse(timestamp=trends["timestamp"], data=trends)
        except Exception as e:
            logger.error(f"Error getting trends: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve trends")


    @app.get("/monitoring/endpoint/{endpoint_name}", response_model=MonitoringResponse)
    async def get_endpoint_metrics(endpoint_name: str):
        """Get metrics for a specific endpoint"""
        try:
            # URL decode the endpoint name
            import urllib.parse

            decoded_endpoint = urllib.parse.unquote(endpoint_name)
            if not decoded_endpoint.startswith("/"):
                decoded_endpoint = "/" + decoded_endpoint

            metrics = business_metrics.get_endpoint_metrics(decoded_endpoint)
            return MonitoringResponse(timestamp=metrics["timestamp"], data=metrics)
        except Exception as e:
            logger.error(f"Error getting endpoint metrics: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve endpoint metrics"
            )

except Exception as e:
    logger.error(f"Failed to register monitoring endpoints: {e}")


@app.get("/")
async def root():
    """Root endpoint with comprehensive API information and health summary"""
    uptime = time.time() - startup_start_time
    model_loaded = model is not None
    feature_engine_loaded = feature_engine is not None

    # Quick health summary
    health_summary = {
        "api_status": (
            "operational" if (model_loaded and feature_engine_loaded) else "degraded"
        ),
        "uptime_seconds": round(uptime, 2),
        "model_ready": model_loaded and feature_engine_loaded,
    }

    # Get basic metrics if available
    try:
        current_metrics = business_metrics.get_current_metrics()["metrics"]
        health_summary.update(
            {
                "total_requests": current_metrics.get("total_requests", 0),
                "requests_per_minute": round(
                    current_metrics.get("requests_per_minute", 0), 2
                ),
                "error_rate_percent": round(
                    current_metrics.get("error_rate_percent", 0), 2
                ),
                "avg_response_time_ms": round(
                    current_metrics.get("avg_response_time_ms", 0), 1
                ),
            }
        )
    except:
        pass

    return {
        "message": "Medical Document Classifier API with Production Monitoring",
        "version": "1.0.0",
        "model_version": "1.0.0",
        "health_summary": health_summary,
        "endpoints": {
            "health": "/health - Detailed health check with monitoring data",
            "predict": "/predict - Make predictions with comprehensive logging",
            "monitoring": {
                "metrics": "/monitoring/metrics - Business metrics and KPIs",
                "performance": "/monitoring/performance - Model performance summary",
                "drift": "/monitoring/drift?window_size=100 - Model drift detection",
                "errors": "/monitoring/errors?hours=24 - Error analysis",
                "anomalies": "/monitoring/anomalies - Real-time anomaly detection",
                "trends": "/monitoring/trends?hours=24 - Hourly performance trends",
                "endpoint_metrics": "/monitoring/endpoint/{endpoint_name} - Per-endpoint metrics",
            },
            "docs": "/docs - Interactive API documentation",
        },
        "monitoring_features": [
            "Real-time prediction logging with drift detection",
            "Business metrics tracking (volume, latency, errors)",
            "Performance monitoring with percentiles",
            "Anomaly detection and alerting",
            "Request-level tracing and debugging",
            "Model staleness monitoring",
            "System resource tracking",
        ],
    }


# Model staleness monitoring utility
def check_model_staleness() -> Dict[str, Any]:
    """Check if the model is getting stale based on usage patterns and age"""
    try:
        current_time = datetime.now(timezone.utc)

        # Model age since loading
        model_loaded_time = datetime.fromisoformat(
            prediction_logger.model_loaded_at.replace("Z", "+00:00")
        )
        model_age_hours = (current_time - model_loaded_time).total_seconds() / 3600

        # Get recent prediction activity
        performance_stats = prediction_logger.get_performance_summary()
        total_predictions = performance_stats["performance_stats"]["total_predictions"]

        # Check for staleness indicators
        staleness_alerts = []

        # Model is very old (> 7 days)
        if model_age_hours > 168:  # 7 days
            staleness_alerts.append(
                {
                    "type": "MODEL_AGE_HIGH",
                    "severity": (
                        "HIGH" if model_age_hours > 720 else "MEDIUM"
                    ),  # 30 days = HIGH
                    "message": f"Model is {model_age_hours:.1f} hours old (threshold: 168 hours)",
                    "value": model_age_hours,
                }
            )

        # Very low prediction volume (potential data drift or service not being used)
        recent_metrics = business_metrics.get_current_metrics()["metrics"]
        requests_per_minute = recent_metrics.get("requests_per_minute", 0)

        if (
            requests_per_minute < 0.1 and model_age_hours > 1
        ):  # Less than 6 requests per hour after 1 hour
            staleness_alerts.append(
                {
                    "type": "LOW_USAGE_VOLUME",
                    "severity": "MEDIUM",
                    "message": f"Very low prediction volume: {requests_per_minute:.2f} req/min",
                    "value": requests_per_minute,
                }
            )

        # Check for confidence degradation over time
        if len(prediction_logger.recent_predictions) >= 50:
            recent_confidences = [
                p["confidence"] for p in prediction_logger.recent_predictions[-50:]
            ]
            avg_recent_confidence = sum(recent_confidences) / len(recent_confidences)

            if avg_recent_confidence < 0.6:
                staleness_alerts.append(
                    {
                        "type": "CONFIDENCE_DEGRADATION",
                        "severity": "HIGH" if avg_recent_confidence < 0.5 else "MEDIUM",
                        "message": f"Average confidence dropping: {avg_recent_confidence:.3f} (threshold: 0.6)",
                        "value": avg_recent_confidence,
                    }
                )

        return {
            "model_age_hours": model_age_hours,
            "model_loaded_at": prediction_logger.model_loaded_at,
            "total_predictions": total_predictions,
            "staleness_alerts": staleness_alerts,
            "staleness_score": len(staleness_alerts),
            "recommendation": _get_staleness_recommendation(staleness_alerts),
            "timestamp": current_time.isoformat(),
        }
    except Exception as e:
        logger.error(f"Error checking model staleness: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def _get_staleness_recommendation(alerts: List[Dict]) -> str:
    """Get recommendation based on staleness alerts"""
    if not alerts:
        return "Model is fresh and performing well"

    high_severity_count = sum(1 for alert in alerts if alert["severity"] == "HIGH")
    medium_severity_count = sum(1 for alert in alerts if alert["severity"] == "MEDIUM")

    if high_severity_count >= 2:
        return "URGENT: Consider retraining model immediately with fresh data"
    elif high_severity_count >= 1:
        return "Consider retraining model with recent data within 24-48 hours"
    elif medium_severity_count >= 2:
        return "Monitor closely and consider retraining within 1 week"
    else:
        return "Monitor model performance and consider retraining schedule"


# Monitoring endpoints wrapped in try-except to ensure they register
try:
    @app.get("/monitoring/staleness", response_model=MonitoringResponse)
    async def get_model_staleness():
        """Get model staleness analysis and recommendations"""
        try:
            staleness_data = check_model_staleness()
            return MonitoringResponse(
                timestamp=staleness_data["timestamp"], data=staleness_data
            )
        except Exception as e:
            logger.error(f"Error getting model staleness: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve staleness analysis"
            )

except Exception as e:
    logger.error(f"Failed to register monitoring/staleness endpoint: {e}")

# DAG Management Endpoints wrapped in try-except
try:
    @app.get("/dags")
    async def list_dags():
        """List all available DAGs and their status"""
        if not dag_scheduler:
            raise HTTPException(status_code=503, detail="DAG scheduler not available")
        
        try:
            dags_info = []
            for dag_id in dag_scheduler.dags:
                dag_status = dag_scheduler.get_dag_status(dag_id)
                dags_info.append(dag_status)
            
            return {
                "dags": dags_info,
                "scheduler_running": dag_scheduler.running,
                "total_dags": len(dag_scheduler.dags)
            }
        except Exception as e:
            logger.error(f"Error listing DAGs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/dags/{dag_id}/status")
    async def get_dag_status(dag_id: str):
        """Get detailed status of a specific DAG"""
        if not dag_scheduler:
            raise HTTPException(status_code=503, detail="DAG scheduler not available")
        
        try:
            return dag_scheduler.get_dag_status(dag_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting DAG status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/dags/{dag_id}/run")
    async def trigger_dag(dag_id: str):
        """Manually trigger a DAG execution"""
        if not dag_scheduler:
            raise HTTPException(status_code=503, detail="DAG scheduler not available")
        
        try:
            result = dag_scheduler.run_dag_now(dag_id)
            return {
                "message": f"DAG {dag_id} triggered successfully",
                "run_result": {
                    "status": result["status"],
                    "duration_seconds": result["duration_seconds"],
                    "success_count": result["success_count"],
                    "failed_count": result["failed_count"],
                    "start_time": result["start_time"].isoformat(),
                    "end_time": result["end_time"].isoformat()
                }
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error running DAG: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/scheduler/start")
    async def start_scheduler():
        """Start the DAG scheduler"""
        if not dag_scheduler:
            raise HTTPException(status_code=503, detail="DAG scheduler not available")
        
        try:
            dag_scheduler.start()
            return {"message": "DAG scheduler started", "running": dag_scheduler.running}
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/scheduler/stop")
    async def stop_scheduler():
        """Stop the DAG scheduler"""
        if not dag_scheduler:
            raise HTTPException(status_code=503, detail="DAG scheduler not available")
        
        try:
            dag_scheduler.stop()
            return {"message": "DAG scheduler stopped", "running": dag_scheduler.running}
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            raise HTTPException(status_code=500, detail=str(e))

except Exception as e:
    logger.error(f"Failed to register DAG management endpoints: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000) 
