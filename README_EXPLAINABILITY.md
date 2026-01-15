# EEG Model Explainability Dashboard

A comprehensive FastAPI + React dashboard for explaining EEG psychiatric disorder classification models using SHAP, LIME, and advanced visualization techniques.

## 🎯 Features

- **Model Predictions**: Get ensemble predictions from KNN, Random Forest, and Extra Trees
- **SHAP Explanations**: Understand which features contribute most to predictions
- **LIME Explanations**: Get local interpretable model-agnostic explanations
- **Feature Space Visualization**: See where the patient lies in PCA-reduced feature space
- **Decision Path Analysis**: Visualize how the model processes the input through decision trees and KNN neighbors

## 📁 Project Structure

```
├── backend/
│   ├── main.py              # FastAPI application
│   ├── explainer.py         # SHAP, LIME, and visualization logic
│   ├── requirements.txt     # Python dependencies
│   └── sample_data.csv      # Sample patient data
├── frontend/
│   ├── index.html           # Main HTML file
│   ├── style.css            # Styling
│   └── app.js               # React application
└── all_models/              # Trained model files (17 disorders)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Modern web browser

### Installation

1. **Install Python dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Start the backend server**:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

3. **Open the frontend**:
```bash
cd ../frontend
python -m http.server 3000
```

Or simply open `frontend/index.html` in your browser.

The dashboard will be available at `http://localhost:3000`

## 📊 How to Use

1. **Select a Disorder**: Choose from 17 psychiatric disorders in the dropdown
2. **Get Prediction**: Click "Predict" to see the ensemble model's prediction
3. **Explain with SHAP**: Click "SHAP Explanation" to see feature importance using SHAP values
4. **Explain with LIME**: Click "LIME Explanation" to get local model-agnostic explanations
5. **Visualize Feature Space**: Click "Feature Space" to see PCA visualization
6. **Analyze Decision Path**: Click "Decision Path" to see how the model makes decisions

## 🔬 Explainability Methods

### SHAP (SHapley Additive exPlanations)
- Shows exact feature contributions to predictions
- Uses TreeExplainer for fast and exact explanations
- Visualizes positive and negative feature impacts

### LIME (Local Interpretable Model-agnostic Explanations)
- Provides local explanations around the prediction
- Model-agnostic approach
- Shows feature ranges that influence the prediction

### Feature Space Visualization
- PCA dimensionality reduction to 2D
- Shows where the patient lies in feature space
- Displays explained variance and principal component contributions

### Decision Path Analysis
- Visualizes tree traversal in Random Forest
- Shows KNN neighbor distances
- Displays feature importance rankings

## 🎨 Sample Patient

The dashboard uses a real sample from the EEG dataset:
- Patient ID: 1
- Sex: Male
- Age: 57
- Actual Disorder: Addictive disorder (Alcohol use disorder)
- Features: 1140 EEG features across 6 frequency bands

## 📡 API Endpoints

- `GET /` - API info
- `GET /disorders` - List available disorders
- `GET /sample` - Get sample patient data
- `POST /predict` - Make prediction
- `POST /explain/shap` - Generate SHAP explanation
- `POST /explain/lime` - Generate LIME explanation
- `POST /visualize/feature_space` - Feature space visualization
- `POST /visualize/decision_path` - Decision path analysis

## 🧪 Testing

Test the API directly:

```bash
# Get disorders
curl http://localhost:8000/disorders

# Get sample data
curl http://localhost:8000/sample

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"disorder": "Mood_disorder"}'
```

## 📝 Notes

- The dashboard uses the sample patient from the first row of the EEG dataset
- All explanations are computed in real-time
- SHAP uses TreeExplainer for fast exact explanations on tree-based models
- LIME uses tabular explainer for local explanations
- Feature space visualization uses PCA (faster than t-SNE/UMAP for single point)

## 🐛 Troubleshooting

**SHAP not working:**
```bash
pip install shap
```

**LIME not working:**
```bash
pip install lime
```

**CORS errors:**
- Make sure both backend and frontend are running
- Backend should be on port 8000
- Frontend should be on a different port (e.g., 3000)

**Models not loading:**
- Ensure `all_models/` directory is in the parent directory
- Check that all model files (.pkl) and metadata (.json) are present

## 🔮 Future Enhancements

- Support for custom patient data input
- Batch prediction and explanation
- Model comparison across disorders
- Interactive feature importance plots
- Export explanations as PDF reports
- Integration with more explainability methods (Anchors, Counterfactuals)

## 📚 References

- SHAP: https://github.com/slundberg/shap
- LIME: https://github.com/marcotcr/lime
- FastAPI: https://fastapi.tiangolo.com/
- React: https://reactjs.org/
- Plotly: https://plotly.com/javascript/
