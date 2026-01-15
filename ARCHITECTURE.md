# 🏗️ Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│                     http://localhost:3000                    │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Predict  │  │  SHAP    │  │  LIME    │  │  Visual  │   │
│  │  Button  │  │  Button  │  │  Button  │  │  Buttons │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │             │          │
│       └─────────────┴──────────────┴─────────────┘          │
│                          │                                   │
│                     HTTP POST                                │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                          │
│                 http://localhost:8000                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                     main.py                            │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │ │
│  │  │ /predict │  │  /shap   │  │  /lime   │            │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘            │ │
│  └───────┼─────────────┼─────────────┼───────────────────┘ │
│          │             │             │                      │
│          ▼             ▼             ▼                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   explainer.py                         │ │
│  │                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐                  │ │
│  │  │ SHAP         │  │ LIME         │                  │ │
│  │  │ TreeExplainer│  │ Tabular      │                  │ │
│  │  └──────────────┘  └──────────────┘                  │ │
│  │                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐                  │ │
│  │  │ PCA          │  │ Decision     │                  │ │
│  │  │ Visualization│  │ Path         │                  │ │
│  │  └──────────────┘  └──────────────┘                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                               │
│                   all_models/                                │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     KNN      │  │ Random Forest│  │ Extra Trees  │     │
│  │  (5 metrics) │  │ (300 trees)  │  │ (300 trees)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                          │                                   │
│                  ┌───────┴────────┐                         │
│                  │   Ensemble     │                         │
│                  │  (F1-weighted) │                         │
│                  └────────────────┘                         │
│                                                              │
│  17 Disorders × (3 models + scaler + metadata)             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Prediction Flow
```
User clicks "Predict"
    ↓
Frontend POST /predict with disorder name
    ↓
Backend loads 3 models + scaler
    ↓
Scale features using StandardScaler
    ↓
Get predictions from KNN, RF, ET
    ↓
Weighted ensemble (using F1-based weights)
    ↓
Apply optimal threshold
    ↓
Return prediction + probabilities + metrics
    ↓
Frontend displays results with confidence
```

### 2. SHAP Flow
```
User clicks "SHAP"
    ↓
Frontend POST /explain/shap
    ↓
Backend creates TreeExplainer for Random Forest
    ↓
Compute SHAP values for all features
    ↓
Get top 20 features by absolute SHAP value
    ↓
Return SHAP values + base value + interpretation
    ↓
Frontend creates bar chart with Plotly
```

### 3. LIME Flow
```
User clicks "LIME"
    ↓
Frontend POST /explain/lime
    ↓
Backend creates LimeTabularExplainer
    ↓
Define ensemble prediction function
    ↓
Explain instance with 20 top features
    ↓
Return feature importances + interpretation
    ↓
Frontend creates bar chart
```

### 4. Feature Space Flow
```
User clicks "Feature Space"
    ↓
Frontend POST /visualize/feature_space
    ↓
Backend applies PCA with 2 components
    ↓
Project sample to 2D space
    ↓
Get explained variance + top contributing features
    ↓
Return PC coordinates + variance + interpretation
    ↓
Frontend creates scatter plot with axes labeled
```

### 5. Decision Path Flow
```
User clicks "Decision Path"
    ↓
Frontend POST /visualize/decision_path
    ↓
Backend extracts decision path from first tree
    ↓
Get node information (feature, threshold, samples)
    ↓
Get KNN neighbor distances
    ↓
Return tree path + neighbor info + feature importance
    ↓
Frontend displays tree traversal + neighbor distances
```

## Component Details

### Frontend Components (React)

| Component | Purpose | State |
|-----------|---------|-------|
| `App` | Main container | disorders, selected, all results, loading, error |
| `SampleInfo` | Display patient metadata | sample data (props) |
| `PredictionCard` | Show prediction results | prediction data (props) |
| `SHAPCard` | SHAP visualization | SHAP data (props) |
| `LIMECard` | LIME visualization | LIME data (props) |
| `FeatureSpaceCard` | PCA visualization | feature space data (props) |
| `DecisionPathCard` | Tree path visualization | decision path data (props) |

### Backend Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/` | GET | - | API info |
| `/disorders` | GET | - | List of 17 disorders with metrics |
| `/sample` | GET | - | Sample patient metadata + features |
| `/predict` | POST | disorder name | Prediction + probabilities + metrics |
| `/explain/shap` | POST | disorder name | SHAP values + top features |
| `/explain/lime` | POST | disorder name | LIME weights + top features |
| `/visualize/feature_space` | POST | disorder name | PCA coordinates + variance |
| `/visualize/decision_path` | POST | disorder name | Tree path + KNN info |

### Model Files (per disorder)

| File | Size | Purpose |
|------|------|---------|
| `{disorder}_knn.pkl` | ~10-13 MB | KNN model (k=5-15, various metrics) |
| `{disorder}_rf.pkl` | ~2-6 MB | Random Forest (300 trees, depth=12) |
| `{disorder}_et.pkl` | ~4-6 MB | Extra Trees (300 trees, depth=12) |
| `{disorder}_scaler.pkl` | ~28 KB | StandardScaler (fitted on training data) |
| `{disorder}_metadata.json` | ~1 KB | Performance metrics + config |
| `{disorder}_feature_importance.json` | ~127 KB | Top 20 features from RF |

## Explainability Methods

### SHAP (Exact Tree Explanations)

**Algorithm**: TreeExplainer
- Computes exact Shapley values for tree-based models
- Polynomial time complexity O(TLD²) where T=trees, L=leaves, D=depth
- Shows additive feature contributions

**Output**:
```python
{
    "base_value": 0.35,           # Expected model output
    "prediction": 0.67,            # base_value + sum(SHAP values)
    "top_features": [
        {
            "feature": "AB.A.delta.a.FP1",
            "shap_value": 0.15,    # Contribution to prediction
            "feature_value": 35.99,
            "importance": 0.15      # Absolute SHAP value
        },
        ...
    ]
}
```

### LIME (Local Surrogate Model)

**Algorithm**: Local Interpretable Model
- Perturbs input features around the instance
- Fits linear model to approximate local behavior
- Model-agnostic (works with any black-box model)

**Output**:
```python
{
    "prediction_probability": 0.67,
    "top_features": [
        {
            "feature": "feature_5",
            "description": "feature_5 > 0.23",  # Rule
            "importance": 0.12,                  # Linear coefficient
            "feature_value": 0.45
        },
        ...
    ]
}
```

### PCA (Feature Space Projection)

**Algorithm**: Principal Component Analysis
- Linear dimensionality reduction
- Projects 1140 features → 2D for visualization
- Preserves maximum variance

**Output**:
```python
{
    "coordinates": {"pc1": 2.34, "pc2": -1.12},
    "explained_variance": {
        "pc1": 0.25,  # 25% of variance
        "pc2": 0.15,  # 15% of variance
        "total": 0.40 # 40% total
    }
}
```

### Decision Path (Tree Traversal)

**Algorithm**: Tree Path Extraction
- Follows decision tree from root to leaf
- Records feature indices, thresholds, directions
- Shows actual model internals

**Output**:
```python
{
    "example_tree_path": [
        {
            "node_id": 0,
            "type": "decision",
            "feature_index": 42,
            "threshold": 0.67,
            "feature_value": 0.84,
            "decision": "right"  # went right because 0.84 > 0.67
        },
        ...
        {
            "node_id": 15,
            "type": "leaf",
            "class_distribution": [10, 45]  # 10 negative, 45 positive
        }
    ]
}
```

## Performance Characteristics

### Explainability Method Speed

| Method | Time | Scalability |
|--------|------|-------------|
| Prediction | ~50ms | O(N × K) for KNN, O(N × T) for RF |
| SHAP | ~200ms | O(T × L × D²) - fast for trees |
| LIME | ~2-5s | O(P × N) where P=perturbations (~5000) |
| PCA | ~100ms | O(N × F²) where F=features (1140) |
| Decision Path | ~100ms | O(T × D) where D=depth (~12) |

### Memory Usage

| Component | Memory |
|-----------|--------|
| Single model set (3 models + scaler) | ~25 MB |
| All 17 disorders | ~425 MB |
| SHAP computation | ~50 MB |
| LIME computation | ~100 MB |
| Frontend state | ~10 MB |

## Security & Best Practices

### Backend
- ✅ CORS enabled for frontend access
- ✅ Pydantic validation for all inputs
- ✅ Error handling with try-catch
- ✅ Model caching to avoid reloading
- ⚠️ No authentication (add for production)
- ⚠️ No rate limiting (add for production)

### Frontend
- ✅ Loading states for async operations
- ✅ Error display with user-friendly messages
- ✅ Responsive design for mobile
- ✅ Modular React components
- ⚠️ No input validation (uses sample only)
- ⚠️ No session management

## Extension Points

### Easy Additions
1. **Multiple patients**: Upload CSV with multiple rows
2. **Custom features**: Form to input EEG features manually
3. **Disorder comparison**: Compare predictions across disorders
4. **Export reports**: Generate PDF with all explanations
5. **Batch processing**: Process multiple patients at once

### Advanced Additions
1. **Real-time EEG**: Connect to EEG device for live predictions
2. **Interactive plots**: Click features to see details
3. **Counterfactual explanations**: Show what would change prediction
4. **Anchor explanations**: Find minimal sufficient feature sets
5. **Model comparison**: Train new models and compare
6. **Confidence calibration**: Improve probability estimates
7. **Uncertainty quantification**: Bayesian approaches for uncertainty

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | React 18 | UI components |
| Visualization | Plotly.js | Interactive charts |
| Styling | Custom CSS | Gradient themes |
| Backend | FastAPI | REST API |
| ML Models | Scikit-learn | KNN, RF, ET |
| Explainability | SHAP, LIME | Model interpretability |
| Dimensionality Reduction | PCA | Feature space visualization |
| Serialization | Joblib, JSON | Model persistence |

## Deployment Considerations

### Local Development (Current)
- Backend: `python main.py` on port 8000
- Frontend: `python -m http.server 3000` or open HTML

### Production Options

**Option 1: Docker**
```dockerfile
# Backend Dockerfile
FROM python:3.9
COPY backend/ /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]

# Frontend: Serve static files via nginx
```

**Option 2: Cloud Hosting**
- Backend: AWS Lambda, Google Cloud Run, Heroku
- Frontend: Netlify, Vercel, GitHub Pages
- Models: S3, Google Cloud Storage

**Option 3: Kubernetes**
- Backend pod with autoscaling
- Frontend served by nginx
- Models in persistent volume

## Maintenance

### Regular Tasks
- Update SHAP/LIME libraries (monthly)
- Retrain models with new data (quarterly)
- Check for security vulnerabilities (weekly)
- Monitor API latency (daily in production)

### Troubleshooting
1. **SHAP slow**: Use KernelExplainer with fewer samples
2. **LIME slow**: Reduce num_samples from 5000 to 1000
3. **Memory issues**: Load models on demand, don't cache
4. **CORS errors**: Check backend is on port 8000

---

**Last Updated**: 2026-01-10
**Version**: 1.0.0
**Author**: AI Lab Team
