from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pickle
import numpy as np
from typing import List, Optional
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Document Classifier API",
    description="ML API for classifying medical documents using trained Multinomial Naive Bayes model",
    version="1.0.0"
)

# Global variables for model and feature engine
model = None
feature_engine = None

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
    """Load the trained model and feature engine"""
    global model, feature_engine
    
    try:
        # Get model paths (assuming they're in the data directory)
        model_path = Path("data/optimized_multinomial_nb.pkl")
        feature_engine_path = Path("data/feature_engine.pkl")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not feature_engine_path.exists():
            raise FileNotFoundError(f"Feature engine file not found: {feature_engine_path}")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        
        # Load feature engine
        with open(feature_engine_path, 'rb') as f:
            feature_engine = pickle.load(f)
        logger.info("Feature engine loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.warning("Failed to load model on startup")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        feature_engine_loaded=feature_engine is not None
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
        prediction = model.predict(text_features)[0]
        
        # Get prediction probabilities for confidence and top predictions
        probabilities = model.predict_proba(text_features)[0]
        
        # Get class names (assuming they're available in the model)
        class_names = model.classes_
        
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
        predicted_class_idx = np.where(class_names == prediction)[0][0]
        confidence = float(probabilities[predicted_class_idx])
        
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