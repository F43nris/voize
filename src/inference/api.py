from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pickle
import numpy as np
from typing import List, Optional
import logging
from pathlib import Path
import os
import sys
import time
import json
from datetime import datetime

# Import the feature engine class and make it available for pickle
from src.inference.feature_engine import MedicalTextFeatureEngine

# Monkey patch for pickle deserialization - make the class available 
# in the module namespace that pickle expects
sys.modules['__main__'].MedicalTextFeatureEngine = MedicalTextFeatureEngine

# Configure structured logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Also add a handler for Cloud Run structured logging
import logging
cloud_handler = logging.StreamHandler()
cloud_handler.setFormatter(logging.Formatter(
    '{"timestamp": "%(asctime)s", "severity": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
))
logger.addHandler(cloud_handler)

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
    if not key.startswith(('_', 'LS_COLORS')):  # Skip private/noisy vars
        logger.info(f"  {key}={value}")

# Initialize FastAPI app
app = FastAPI(
    title="Medical Document Classifier API",
    description="ML API for classifying medical documents using trained Multinomial Naive Bayes model",
    version="1.0.0"
)

# Global variables for model and feature engine
model = None
feature_engine = None
startup_start_time = time.time()
model_load_status = {
    "started": False,
    "completed": False,
    "error": None,
    "attempts": 0,
    "last_attempt_time": None,
    "load_duration": None
}

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    top_predictions: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_engine_loaded: bool

def load_model():
    """Load the trained model and feature engine with retry logic and detailed logging"""
    global model, feature_engine, model_load_status
    
    load_start_time = time.time()
    model_load_status["started"] = True
    model_load_status["last_attempt_time"] = datetime.now().isoformat()
    model_load_status["attempts"] += 1
    
    logger.info("🔄 Starting model loading process...")
    logger.info(f"Attempt #{model_load_status['attempts']}")
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        attempt_start_time = time.time()
        try:
            logger.info(f"📁 ATTEMPT {attempt + 1}/{max_retries} - File system inspection")
            
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
                logger.info(f"Total data directory size: {total_size:,} bytes ({total_size/(1024*1024):.2f} MB)")
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
                raise FileNotFoundError(f"Model file not found: {model_path.absolute()}")
            else:
                model_size = model_path.stat().st_size
                logger.info(f"✅ Model file found: {model_size:,} bytes ({model_size/(1024*1024):.2f} MB)")
                
            if not feature_engine_path.exists():
                logger.error(f"❌ Feature engine file not found: {feature_engine_path.absolute()}")
                raise FileNotFoundError(f"Feature engine file not found: {feature_engine_path.absolute()}")
            else:
                fe_size = feature_engine_path.stat().st_size
                logger.info(f"✅ Feature engine file found: {fe_size:,} bytes ({fe_size/(1024*1024):.2f} MB)")
            
            # Load model with timing
            logger.info("📦 Loading model...")
            model_load_start = time.time()
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            model_load_time = time.time() - model_load_start
            logger.info(f"✅ Model loaded successfully in {model_load_time:.2f} seconds")
            logger.info(f"   Model type: {type(model)}")
            if hasattr(model, 'classes_'):
                logger.info(f"   Model classes: {len(model.classes_)} classes")
            
            # Load feature engine with timing
            logger.info("🔧 Loading feature engine...")
            fe_load_start = time.time()
            with open(feature_engine_path, 'rb') as f:
                feature_engine = pickle.load(f)
            fe_load_time = time.time() - fe_load_start
            logger.info(f"✅ Feature engine loaded successfully in {fe_load_time:.2f} seconds")
            logger.info(f"   Feature engine type: {type(feature_engine)}")
            if hasattr(feature_engine, 'label_encoder'):
                if hasattr(feature_engine.label_encoder, 'classes_'):
                    logger.info(f"   Label encoder classes: {len(feature_engine.label_encoder.classes_)}")
            
            # Test the models quickly
            logger.info("🧪 Testing models with sample data...")
            test_text = ["Patient presents with chest pain"]
            try:
                test_features = feature_engine.transform(test_text)
                test_pred = model.predict(test_features)
                test_proba = model.predict_proba(test_features)
                logger.info(f"✅ Model test successful - prediction shape: {test_pred.shape}, proba shape: {test_proba.shape}")
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
    """Load model on startup with detailed logging"""
    startup_time = time.time() - startup_start_time
    logger.info("🚀 FastAPI STARTUP EVENT TRIGGERED")
    logger.info(f"Time since import: {startup_time:.2f} seconds")
    
    logger.info("📊 System information:")
    import psutil
    try:
        memory_info = psutil.virtual_memory()
        logger.info(f"  Available memory: {memory_info.available / (1024**3):.2f} GB")
        logger.info(f"  Total memory: {memory_info.total / (1024**3):.2f} GB")
        logger.info(f"  Memory usage: {memory_info.percent}%")
        
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"  CPU cores: {cpu_count}")
        logger.info(f"  CPU usage: {cpu_percent}%")
    except ImportError:
        logger.warning("psutil not available - cannot show system info")
    except Exception as e:
        logger.warning(f"Error getting system info: {e}")
    
    logger.info("🔄 Starting model loading from startup event...")
    success = load_model()
    
    if success:
        logger.info("✅ STARTUP COMPLETED SUCCESSFULLY!")
    else:
        logger.error("❌ STARTUP FAILED - Model loading unsuccessful")
        # Don't exit here - let health checks handle the failure

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed status information"""
    check_start_time = time.time()
    logger.info("🏥 Health check requested")
    
    model_loaded = model is not None
    feature_engine_loaded = feature_engine is not None
    
    # Get detailed status
    uptime = time.time() - startup_start_time
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(uptime, 2),
        "model_loaded": model_loaded,
        "feature_engine_loaded": feature_engine_loaded,
        "load_status": dict(model_load_status),  # Copy the status
        "system_info": {}
    }
    
    # Add system info if available
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        health_status["system_info"] = {
            "memory_available_gb": round(memory_info.available / (1024**3), 2),
            "memory_percent": memory_info.percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1)
        }
    except:
        pass
    
    logger.info(f"Health status: {json.dumps(health_status, indent=2)}")
    
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
        return {
            "status": "healthy",
            **health_status
        }
    else:
        # Still loading or failed to load
        logger.warning(f"⚠️  Health check FAILED in {check_time:.3f}s - models not ready")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "message": "Models are not loaded or failed to load",
                **health_status
            }
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction on the input text"""
    
    # Check if model is loaded
    if model is None or feature_engine is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Transform text using feature engine
        text_features = feature_engine.transform([request.text])
        
        # Make prediction
        prediction_encoded = model.predict(text_features)[0]
        
        # Decode the prediction back to class name
        prediction = feature_engine.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities for confidence and top predictions
        probabilities = model.predict_proba(text_features)[0]
        
        # Get class names from label encoder
        class_names = feature_engine.label_encoder.classes_
        
        # Create top predictions
        top_indices = np.argsort(probabilities)[-5:][::-1]  # Top 5 predictions
        top_predictions = [
            {
                "class": class_names[idx],
                "confidence": float(probabilities[idx])
            }
            for idx in top_indices
        ]
        
        # Get confidence for the predicted class
        confidence = float(probabilities[prediction_encoded])
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            top_predictions=top_predictions
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Document Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 