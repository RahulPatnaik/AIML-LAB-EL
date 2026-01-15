# Architecture Diagrams

Three simple diagrams for research paper presentation:

## 1. architecture.mmd
**System Architecture (Overall Flow)**
- Input: Patient EEG data (1140 features)
- Processing: Feature scaling → 17 classifiers → Explainability
- Output: Predictions + Visualizations + AI interpretation

Clean 3-layer architecture suitable for papers.

## 2. app-architecture.mmd
**Application Architecture**
- Frontend: React UI + Plotly visualizations
- Backend: FastAPI + Model layer + AI module
- Data: EEG dataset + patient samples

Simple component diagram.

## 3. model-architecture.mmd
**Novel Ensemble Model**
- Input: 1140 EEG features across 6 frequency bands
- Ensemble: KNN + Random Forest + Extra Trees
- Weighted voting with F1-based weights
- Optimal threshold from precision-recall curve
- Output: Binary prediction + probability + confidence

Highlights the novel contribution.

## How to View

### Online
Go to https://mermaid.live and paste any .mmd file contents

### In Research Papers
Export as PNG/SVG from mermaid.live for inclusion in papers

### On GitHub
GitHub renders Mermaid diagrams automatically
