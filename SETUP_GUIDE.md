# 🚀 Quick Setup Guide

## Installation (5 minutes)

### Step 1: Install Backend Dependencies

```bash
cd "AIML LAB EL/backend"
pip install -r requirements.txt
```

This installs:
- FastAPI (web framework)
- Pandas, NumPy, Scikit-learn (data processing)
- SHAP & LIME (explainability)
- Matplotlib, Seaborn (visualization)

### Step 2: Start the Backend Server

```bash
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Keep this terminal open!

### Step 3: Open the Frontend

Open a **new terminal** and run:

```bash
cd "AIML LAB EL/frontend"
python3 -m http.server 3000
```

Or simply open `frontend/index.html` directly in your browser.

### Step 4: Access the Dashboard

Open your browser and go to:
**http://localhost:3000**

## 🎯 Quick Test

1. Select a disorder from the dropdown (e.g., "Mood disorder")
2. Click "🎯 Predict" button
3. You should see prediction results appear!
4. Try "📊 SHAP Explanation" to see feature importance
5. Try "🔍 LIME Explanation" for alternative explanations

## 📁 What You Built

```
AIML LAB EL/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── explainer.py               # SHAP, LIME, visualizations
│   ├── requirements.txt           # Python packages
│   └── sample_data.csv            # Sample patient (Male, 57, Alcohol use disorder)
│
├── frontend/
│   ├── index.html                 # Main page
│   ├── app.js                     # React dashboard
│   └── style.css                  # Styling
│
├── all_models/                    # Your trained models (17 disorders)
│   ├── Mood_disorder_knn.pkl
│   ├── Mood_disorder_rf.pkl
│   ├── Mood_disorder_et.pkl
│   ├── Mood_disorder_scaler.pkl
│   ├── Mood_disorder_metadata.json
│   └── ... (5 more files per disorder × 17 disorders)
│
└── README_EXPLAINABILITY.md       # Full documentation
```

## 🎨 Features Overview

### 1. **Prediction**
- Ensemble of KNN + Random Forest + Extra Trees
- Shows individual model predictions
- Confidence levels (High/Medium/Low)
- Performance metrics (F1, Recall, Precision)

### 2. **SHAP Explanation**
- Tree-based exact explanations
- Shows which features increase/decrease prediction
- Bar chart of top 15 features
- Base value + feature contributions

### 3. **LIME Explanation**
- Local model-agnostic explanations
- Feature ranges that influence prediction
- Alternative perspective to SHAP
- Bar chart of feature weights

### 4. **Feature Space Visualization**
- PCA 2D projection of 1140 features
- Shows where patient lies in feature space
- Displays explained variance
- Top contributing features to each PC

### 5. **Decision Path Analysis**
- Random Forest tree traversal visualization
- KNN neighbor distances
- Feature importance rankings
- Shows decision nodes and thresholds

## 🔧 Troubleshooting

### Backend won't start:
```bash
# Check Python version (needs 3.8+)
python --version

# Reinstall dependencies
pip install -r backend/requirements.txt --upgrade
```

### Models not found:
```bash
# Verify models directory exists
ls all_models/*.pkl | wc -l
# Should show ~85 files (5 files × 17 disorders)
```

### SHAP/LIME errors:
```bash
# Install specific versions
pip install shap==0.42.0 lime==0.2.0.1
```

### Frontend not loading:
- Check that backend is running on port 8000
- Check browser console for errors (F12)
- Try opening `index.html` directly instead of using Python server

### CORS errors:
- Ensure backend is on port 8000
- Ensure frontend is on different port (3000) or opened as file://
- Check browser console for specific CORS error messages

## 💡 Usage Tips

1. **Start with Prediction**: Always click "Predict" first to see the base prediction
2. **Compare SHAP vs LIME**: Different methods may highlight different features
3. **Feature Space**: Helps understand if patient is typical or outlier
4. **Decision Path**: Shows the actual model internals
5. **Sample Patient**: The dashboard uses Patient #1 (Male, 57, Alcohol use disorder)

## 📊 Understanding the Results

### Prediction Output:
- **Probability**: Model's confidence (0-100%)
- **Threshold**: Decision boundary (typically 40-55%)
- **Confidence**: Distance from 0.5 (High if far, Low if near)

### SHAP Values:
- **Positive values**: Feature pushes prediction towards positive class
- **Negative values**: Feature pushes prediction towards negative class
- **Magnitude**: How strongly the feature influences prediction

### LIME Weights:
- Similar to SHAP but computed differently
- Shows feature ranges (e.g., "feature_5 > 0.23")
- Local explanation around the specific patient

### Feature Space (PCA):
- **PC1 & PC2**: Two main directions of variance
- **Explained variance**: How much info is retained (typically 20-40%)
- **Coordinates**: Where patient falls in 2D space

## 🎓 Next Steps

1. **Modify Sample**: Edit `backend/sample_data.csv` to test other patients
2. **Add Disorders**: Train models for more disorders and add to `all_models/`
3. **Enhance Visualizations**: Add 3D plots, heatmaps, confusion matrices
4. **Export Reports**: Add PDF generation for explanations
5. **Real-time Input**: Create form to input custom EEG features

## 📚 Learn More

- **SHAP**: https://github.com/slundberg/shap
- **LIME**: https://github.com/marcotcr/lime
- **FastAPI**: https://fastapi.tiangolo.com/
- **Model Training**: See `train.py` for how models were built

## ✨ You Did It!

You now have a fully functional AI explainability dashboard that:
- Makes predictions on EEG data
- Explains predictions using SHAP and LIME
- Visualizes feature space and decision paths
- Has a beautiful, responsive UI
- Is modular and easy to extend

**Congratulations! 🎉**
