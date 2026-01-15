"""
FastAPI Backend for EEG Model Explainability
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from explainer import ModelExplainer
from ai_explainer import (
    generate_prediction_explanation,
    generate_shap_explanation,
    generate_tsne_explanation,
    generate_umap_explanation,
    generate_pca_explanation,
    generate_lime_explanation,
    generate_decision_path_explanation,
    generate_full_check_explanation
)

app = FastAPI(title="EEG Model Explainability Dashboard")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "all_models"
SAMPLE_CSV = BASE_DIR / "backend" / "sample_data.csv"
PATIENT_SAMPLES = BASE_DIR / "backend" / "patient_samples.json"

# Load sample data
sample_df = pd.read_csv(SAMPLE_CSV)
sample_row = sample_df.iloc[0]

# Load patient samples
with open(PATIENT_SAMPLES, "r") as f:
    patient_samples = json.load(f)

# Feature columns
metadata_cols = ['no.', 'sex', 'age', 'eeg.date', 'education', 'IQ',
                 'main.disorder', 'specific.disorder']
unnamed_cols = [col for col in sample_df.columns if 'Unnamed' in col]
feature_cols = [col for col in sample_df.columns if col not in metadata_cols + unnamed_cols]
sample_features = sample_row[feature_cols].values.astype(float)

# Load available models
with open(MODELS_DIR / "deployment_manifest.json", "r") as f:
    manifest = json.load(f)

available_disorders = [m["safe_name"] for m in manifest["models"]]
explainer_cache = {}


class PredictionRequest(BaseModel):
    features: Optional[List[float]] = None
    disorder: str
    patient_id: Optional[int] = None


class FullCheckRequest(BaseModel):
    patient_id: Optional[int] = None
    features: Optional[List[float]] = None


@app.get("/")
def root():
    return {
        "message": "EEG Model Explainability API",
        "version": "1.0.0",
        "endpoints": [
            "/disorders",
            "/predict",
            "/explain/shap",
            "/explain/lime",
            "/visualize/feature_space",
            "/visualize/tsne",
            "/visualize/umap",
            "/visualize/decision_path",
            "/sample"
        ]
    }


@app.get("/disorders")
def get_disorders():
    """Get list of available disorders"""
    disorders = []
    for model_info in manifest["models"]:
        disorders.append({
            "name": model_info["disorder"],
            "safe_name": model_info["safe_name"],
            "f1_score": model_info["performance"]["f1_score"],
            "recall": model_info["performance"]["recall"],
            "precision": model_info["performance"]["precision"]
        })
    return {"disorders": disorders}


@app.get("/sample")
def get_sample():
    """Get sample patient data"""
    return {
        "metadata": {
            "no": int(sample_row["no."]) if "no." in sample_row else None,
            "sex": str(sample_row["sex"]) if "sex" in sample_row else None,
            "age": int(sample_row["age"]) if "age" in sample_row and pd.notna(sample_row["age"]) else None,
            "main_disorder": str(sample_row["main.disorder"]) if "main.disorder" in sample_row else None,
            "specific_disorder": str(sample_row["specific.disorder"]) if "specific.disorder" in sample_row else None,
        },
        "features": sample_features.tolist(),
        "feature_names": feature_cols,
        "num_features": len(feature_cols)
    }


@app.get("/patients")
def get_patients():
    """Get list of available patient samples"""
    patients = []
    for p in patient_samples:
        patients.append({
            "id": p["id"],
            "sex": p["sex"],
            "age": p["age"],
            "education": p["education"],
            "iq": p["iq"],
            "main_disorder": p["main_disorder"],
            "specific_disorder": p["specific_disorder"],
            "label": f"Patient #{p['id']} - {p['sex']}, Age {p['age']}, {p['specific_disorder']}"
        })
    return {"patients": patients}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Make prediction for a disorder"""
    disorder = request.disorder

    if disorder not in available_disorders:
        raise HTTPException(status_code=404, detail=f"Disorder '{disorder}' not found")

    # Get patient data
    patient_metadata = "Sample patient, Male, 57 years old, Alcohol use disorder"
    if request.patient_id is not None:
        # Find patient by ID
        patient = next((p for p in patient_samples if p["id"] == request.patient_id), None)
        if patient:
            features = np.array(patient["features"]).reshape(1, -1)
            patient_metadata = f"Patient #{patient['id']}: {patient['sex']}, Age {patient['age']}, {patient['specific_disorder']}"
        else:
            features = np.array(request.features if request.features else sample_features).reshape(1, -1)
    else:
        features = np.array(request.features if request.features else sample_features).reshape(1, -1)

    # Get explainer
    if disorder not in explainer_cache:
        explainer_cache[disorder] = ModelExplainer(str(MODELS_DIR), disorder)

    explainer = explainer_cache[disorder]

    # Make prediction
    prediction = explainer.predict(features)

    # Generate AI explanation
    ai_explanation = generate_prediction_explanation(disorder, prediction, patient_metadata)

    return {
        "disorder": disorder,
        "prediction": prediction,
        "ai_explanation": ai_explanation,
        "patient_metadata": patient_metadata
    }


@app.post("/explain/shap")
def explain_shap(request: PredictionRequest):
    """Generate SHAP explanation"""
    disorder = request.disorder

    if disorder not in available_disorders:
        raise HTTPException(status_code=404, detail=f"Disorder '{disorder}' not found")

    # Use provided features or sample
    features = np.array(request.features if request.features else sample_features).reshape(1, -1)

    # Get explainer
    if disorder not in explainer_cache:
        explainer_cache[disorder] = ModelExplainer(str(MODELS_DIR), disorder)

    explainer = explainer_cache[disorder]

    # Generate SHAP values
    shap_explanation = explainer.explain_shap(features)

    # Generate AI explanation
    patient_metadata = f"Sample patient, Male, 57 years old, Alcohol use disorder"
    ai_explanation = generate_shap_explanation(disorder, shap_explanation, patient_metadata)

    return {
        "disorder": disorder,
        "shap_values": shap_explanation,
        "ai_explanation": ai_explanation
    }


@app.post("/explain/lime")
def explain_lime(request: PredictionRequest):
    """Generate LIME explanation"""
    disorder = request.disorder

    if disorder not in available_disorders:
        raise HTTPException(status_code=404, detail=f"Disorder '{disorder}' not found")

    # Use provided features or sample
    features = np.array(request.features if request.features else sample_features).reshape(1, -1)

    # Get explainer
    if disorder not in explainer_cache:
        explainer_cache[disorder] = ModelExplainer(str(MODELS_DIR), disorder)

    explainer = explainer_cache[disorder]

    # Generate LIME explanation
    lime_explanation = explainer.explain_lime(features)

    # Generate AI explanation
    patient_metadata = f"Sample patient, Male, 57 years old, Alcohol use disorder"
    ai_explanation = generate_lime_explanation(disorder, lime_explanation, patient_metadata)

    return {
        "disorder": disorder,
        "lime_values": lime_explanation,
        "ai_explanation": ai_explanation
    }


@app.post("/visualize/feature_space")
def visualize_feature_space(request: PredictionRequest):
    """Generate feature space visualization (PCA/t-SNE/UMAP)"""
    disorder = request.disorder

    if disorder not in available_disorders:
        raise HTTPException(status_code=404, detail=f"Disorder '{disorder}' not found")

    # Use provided features or sample
    features = np.array(request.features if request.features else sample_features).reshape(1, -1)

    # Get explainer
    if disorder not in explainer_cache:
        explainer_cache[disorder] = ModelExplainer(str(MODELS_DIR), disorder)

    explainer = explainer_cache[disorder]

    # Generate visualization
    visualization = explainer.visualize_feature_space(features)

    # Generate AI explanation
    patient_metadata = f"Sample patient, Male, 57 years old, Alcohol use disorder"
    ai_explanation = generate_pca_explanation(disorder, visualization, patient_metadata)

    return {
        "disorder": disorder,
        "visualization": visualization,
        "ai_explanation": ai_explanation
    }


@app.post("/visualize/decision_path")
def visualize_decision_path(request: PredictionRequest):
    """Visualize decision path through the model"""
    disorder = request.disorder

    if disorder not in available_disorders:
        raise HTTPException(status_code=404, detail=f"Disorder '{disorder}' not found")

    # Use provided features or sample
    features = np.array(request.features if request.features else sample_features).reshape(1, -1)

    # Get explainer
    if disorder not in explainer_cache:
        explainer_cache[disorder] = ModelExplainer(str(MODELS_DIR), disorder)

    explainer = explainer_cache[disorder]

    # Generate decision path
    decision_path = explainer.visualize_decision_path(features)

    # Generate AI explanation
    patient_metadata = f"Sample patient, Male, 57 years old, Alcohol use disorder"
    ai_explanation = generate_decision_path_explanation(disorder, decision_path, patient_metadata)

    return {
        "disorder": disorder,
        "decision_path": decision_path,
        "ai_explanation": ai_explanation
    }


@app.post("/visualize/tsne")
def visualize_tsne(request: PredictionRequest):
    """Generate t-SNE visualization"""
    disorder = request.disorder

    if disorder not in available_disorders:
        raise HTTPException(status_code=404, detail=f"Disorder '{disorder}' not found")

    # Use provided features or sample
    features = np.array(request.features if request.features else sample_features).reshape(1, -1)

    # Get explainer
    if disorder not in explainer_cache:
        explainer_cache[disorder] = ModelExplainer(str(MODELS_DIR), disorder)

    explainer = explainer_cache[disorder]

    # Generate t-SNE visualization
    tsne_viz = explainer.visualize_tsne(features)

    # Generate AI explanation
    patient_metadata = f"Sample patient, Male, 57 years old, Alcohol use disorder"
    ai_explanation = generate_tsne_explanation(disorder, tsne_viz, patient_metadata)

    return {
        "disorder": disorder,
        "tsne": tsne_viz,
        "ai_explanation": ai_explanation
    }


@app.post("/visualize/umap")
def visualize_umap(request: PredictionRequest):
    """Generate UMAP visualization"""
    disorder = request.disorder

    if disorder not in available_disorders:
        raise HTTPException(status_code=404, detail=f"Disorder '{disorder}' not found")

    # Use provided features or sample
    features = np.array(request.features if request.features else sample_features).reshape(1, -1)

    # Get explainer
    if disorder not in explainer_cache:
        explainer_cache[disorder] = ModelExplainer(str(MODELS_DIR), disorder)

    explainer = explainer_cache[disorder]

    # Generate UMAP visualization
    umap_viz = explainer.visualize_umap(features)

    # Generate AI explanation
    patient_metadata = f"Sample patient, Male, 57 years old, Alcohol use disorder"
    ai_explanation = generate_umap_explanation(disorder, umap_viz, patient_metadata)

    return {
        "disorder": disorder,
        "umap": umap_viz,
        "ai_explanation": ai_explanation
    }


@app.post("/predict/full_check")
def full_check(request: FullCheckRequest):
    """Run prediction on ALL disorders and provide comprehensive analysis"""

    # Get patient data
    patient_metadata = "Sample patient, Male, 57 years old, Alcohol use disorder"
    if request.patient_id is not None:
        # Find patient by ID
        patient = next((p for p in patient_samples if p["id"] == request.patient_id), None)
        if patient:
            features = np.array(patient["features"]).reshape(1, -1)
            patient_metadata = f"Patient #{patient['id']}: {patient['sex']}, Age {patient['age']}, {patient['specific_disorder']}"
        else:
            features = np.array(request.features if request.features else sample_features).reshape(1, -1)
    else:
        features = np.array(request.features if request.features else sample_features).reshape(1, -1)

    # Run predictions on all disorders
    all_predictions = []

    for disorder in available_disorders:
        # Get explainer
        if disorder not in explainer_cache:
            explainer_cache[disorder] = ModelExplainer(str(MODELS_DIR), disorder)

        explainer = explainer_cache[disorder]

        # Make prediction
        prediction = explainer.predict(features)

        all_predictions.append({
            "disorder": disorder,
            "prediction": prediction["prediction"],
            "probability": prediction["probability"],
            "confidence": prediction["confidence"],
            "threshold": prediction["threshold"],
            "individual_predictions": prediction["individual_predictions"]
        })

    # Sort by probability (descending)
    all_predictions.sort(key=lambda x: x["probability"], reverse=True)

    # Generate comprehensive AI analysis
    ai_explanation = generate_full_check_explanation(all_predictions, patient_metadata)

    # Get top 5 most likely disorders
    top_disorders = all_predictions[:5]

    return {
        "patient_metadata": patient_metadata,
        "all_predictions": all_predictions,
        "top_disorders": top_disorders,
        "total_checked": len(all_predictions),
        "ai_explanation": ai_explanation
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
