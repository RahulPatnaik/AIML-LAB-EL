"""
FastAPI Deployment Template
Load saved models and serve predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
from typing import List, Dict

app = FastAPI(title="EEG Psychiatric Disorder Classifier API")

# Load all models on startup
models = {}

def load_models():
    """Load all trained models"""
    with open('deployment_manifest.json', 'r') as f:
        manifest = json.load(f)
    
    for model_info in manifest['models']:
        safe_name = model_info['safe_name']
        models[safe_name] = {
            'knn': joblib.load(f"{safe_name}_knn.pkl"),
            'rf': joblib.load(f"{safe_name}_rf.pkl"),
            'et': joblib.load(f"{safe_name}_et.pkl"),
            'scaler': joblib.load(f"{safe_name}_scaler.pkl"),
            'metadata': json.load(open(f"{safe_name}_metadata.json")),
            'feature_importance': json.load(open(f"{safe_name}_feature_importance.json"))
        }
    
    return models

# Load models
models = load_models()

class EEGFeatures(BaseModel):
    features: List[float]  # 1140 features

class PredictionResponse(BaseModel):
    disorder: str
    prediction: int
    probability: float
    confidence: str
    threshold: float
    top_features: List[Dict]

@app.get("/")
def root():
    return {
        "message": "EEG Psychiatric Disorder Classifier API",
        "available_disorders": list(models.keys()),
        "total_models": len(models)
    }

@app.get("/models")
def list_models():
    """List all available models with performance metrics"""
    return {
        disorder: {
            "f1_score": models[disorder]['metadata']['performance_metrics']['f1_score'],
            "recall": models[disorder]['metadata']['performance_metrics']['recall'],
            "precision": models[disorder]['metadata']['performance_metrics']['precision']
        }
        for disorder in models.keys()
    }

@app.post("/predict/{disorder_name}", response_model=PredictionResponse)
def predict(disorder_name: str, features: EEGFeatures):
    """Make prediction for a specific disorder"""
    
    if disorder_name not in models:
        raise HTTPException(status_code=404, detail=f"Model for {disorder_name} not found")
    
    # Get model
    model = models[disorder_name]
    
    # Validate input
    if len(features.features) != 1140:
        raise HTTPException(status_code=400, detail=f"Expected 1140 features, got {len(features.features)}")
    
    # Preprocess
    X = np.array(features.features).reshape(1, -1)
    X_scaled = model['scaler'].transform(X)
    
    # Get predictions from all models
    knn_proba = model['knn'].predict_proba(X_scaled)[0, 1]
    rf_proba = model['rf'].predict_proba(X_scaled)[0, 1]
    et_proba = model['et'].predict_proba(X_scaled)[0, 1]
    
    # Ensemble (F1-Weighted)
    weights = model['metadata']['model_config']['ensemble_weights']
    ensemble_proba = (weights['knn'] * knn_proba + 
                     weights['rf'] * rf_proba + 
                     weights['et'] * et_proba)
    
    # Apply threshold
    threshold = model['metadata']['model_config']['optimal_threshold']
    prediction = 1 if ensemble_proba >= threshold else 0
    
    # Confidence
    if abs(ensemble_proba - 0.5) > 0.3:
        confidence = "High"
    elif abs(ensemble_proba - 0.5) > 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Top features
    top_features = model['feature_importance']['top_20_features'][:5]
    
    return PredictionResponse(
        disorder=model['metadata']['disorder'],
        prediction=prediction,
        probability=float(ensemble_proba),
        confidence=confidence,
        threshold=float(threshold),
        top_features=top_features
    )

@app.get("/explain/{disorder_name}")
def explain_model(disorder_name: str):
    """Get model explanation and feature importance"""
    
    if disorder_name not in models:
        raise HTTPException(status_code=404, detail=f"Model for {disorder_name} not found")
    
    model = models[disorder_name]
    
    return {
        "disorder": model['metadata']['disorder'],
        "performance": model['metadata']['performance_metrics'],
        "clinical_metrics": model['metadata']['clinical_metrics'],
        "top_features": model['feature_importance']['top_20_features'],
        "model_config": model['metadata']['model_config']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
