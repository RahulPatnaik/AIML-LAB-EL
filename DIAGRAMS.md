# Architecture Diagrams

This project includes 4 Mermaid diagrams visualizing the system architecture:

## 1. architecture.mmd
**Complete System Architecture**
- Frontend components (React + Plotly)
- Backend API endpoints (FastAPI)
- Model layer with ensemble approach
- Explainability modules (SHAP, LIME, PCA, t-SNE, UMAP)
- AI integration (Mistral API)
- Data sources

View: Paste contents into https://mermaid.live

## 2. dataflow.mmd
**User Data Flow**
- Input: Patient selection + disorder selection
- Processing: Feature scaling → Ensemble prediction → Explanation generation → AI interpretation
- Output: Results + visualizations + AI text

View: Paste contents into https://mermaid.live

## 3. model-architecture.mmd
**Model Architecture Details**
- Input: 1140 EEG features across 6 frequency bands
- 17 disorder classifiers
- Each uses 3-model ensemble (KNN + RF + ET)
- Weighted voting with optimal thresholds
- Explainability outputs

View: Paste contents into https://mermaid.live

## 4. full-check-flow.mmd
**Full Check Feature Flow**
- Parallel execution across all 17 disorders
- Result aggregation and sorting
- AI-powered comprehensive analysis
- Display with top 5 highlights + complete table

View: Paste contents into https://mermaid.live

## How to View

### Online (Recommended)
1. Go to https://mermaid.live
2. Copy contents of any .mmd file
3. Paste into the editor

### In Markdown
Many markdown viewers support Mermaid:
- GitHub (renders automatically)
- VS Code (with Mermaid extension)
- Notion, Obsidian, etc.

### Export as Image
Use mermaid.live to export as PNG/SVG for presentations
