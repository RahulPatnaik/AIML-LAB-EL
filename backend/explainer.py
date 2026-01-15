"""
Model Explainability Module
Implements SHAP, LIME, and visualization techniques
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Try importing SHAP and LIME (graceful fallback if not installed)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  LIME not available. Install with: pip install lime")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP not available. Install with: pip install umap-learn")


class ModelExplainer:
    """Explainer for ensemble EEG psychiatric disorder models"""

    def __init__(self, models_dir: str, disorder: str):
        """
        Initialize explainer for a specific disorder

        Args:
            models_dir: Path to directory containing model files
            disorder: Safe name of the disorder (e.g., "Mood_disorder")
        """
        self.models_dir = Path(models_dir)
        self.disorder = disorder

        # Load models
        self.knn_model = joblib.load(self.models_dir / f"{disorder}_knn.pkl")
        self.rf_model = joblib.load(self.models_dir / f"{disorder}_rf.pkl")
        self.et_model = joblib.load(self.models_dir / f"{disorder}_et.pkl")
        self.scaler = joblib.load(self.models_dir / f"{disorder}_scaler.pkl")

        # Load metadata
        with open(self.models_dir / f"{disorder}_metadata.json", "r") as f:
            self.metadata = json.load(f)

        # Load feature importance
        with open(self.models_dir / f"{disorder}_feature_importance.json", "r") as f:
            self.feature_importance = json.load(f)

        # Ensemble weights
        self.ensemble_weights = self.metadata["model_config"]["ensemble_weights"]
        self.threshold = self.metadata["model_config"]["optimal_threshold"]

        # Feature names (will be set when data is loaded)
        self.feature_names = None
        self.background_data = None

    def _ensemble_predict_proba(self, X):
        """Internal ensemble prediction"""
        knn_proba = self.knn_model.predict_proba(X)[:, 1]
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        et_proba = self.et_model.predict_proba(X)[:, 1]

        # Weighted ensemble
        ensemble_proba = (
            self.ensemble_weights["knn"] * knn_proba +
            self.ensemble_weights["rf"] * rf_proba +
            self.ensemble_weights["et"] * et_proba
        )
        return ensemble_proba

    def predict(self, X):
        """
        Make prediction

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Dict with prediction results
        """
        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get individual model predictions
        knn_proba = self.knn_model.predict_proba(X_scaled)[0, 1]
        rf_proba = self.rf_model.predict_proba(X_scaled)[0, 1]
        et_proba = self.et_model.predict_proba(X_scaled)[0, 1]

        # Ensemble prediction
        ensemble_proba = self._ensemble_predict_proba(X_scaled)[0]

        # Apply threshold
        prediction = 1 if ensemble_proba >= self.threshold else 0

        # Confidence
        distance_from_threshold = abs(ensemble_proba - 0.5)
        if distance_from_threshold > 0.3:
            confidence = "High"
        elif distance_from_threshold > 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "prediction": int(prediction),
            "probability": float(ensemble_proba),
            "threshold": float(self.threshold),
            "confidence": confidence,
            "individual_predictions": {
                "knn": float(knn_proba),
                "random_forest": float(rf_proba),
                "extra_trees": float(et_proba)
            },
            "ensemble_weights": self.ensemble_weights,
            "performance_metrics": self.metadata["performance_metrics"]
        }

    def explain_shap(self, X):
        """
        Generate SHAP explanations

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Dict with SHAP values and interpretations
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not installed. Run: pip install shap"}

        X_scaled = self.scaler.transform(X)

        # Use TreeExplainer for RF and ET (fast and exact for trees)
        try:
            # Explain Random Forest (usually the most important model)
            explainer_rf = shap.TreeExplainer(self.rf_model)
            shap_values_rf = explainer_rf.shap_values(X_scaled)

            # For binary classification, take positive class
            if isinstance(shap_values_rf, list):
                shap_values_rf = shap_values_rf[1]

            # Ensure it's a 2D array
            if len(shap_values_rf.shape) == 1:
                shap_values_rf = shap_values_rf.reshape(1, -1)

            # Get top features - convert arrays to Python lists FIRST
            shap_abs = np.abs(shap_values_rf[0])

            # Convert all arrays to Python lists to avoid numpy scalar issues
            shap_values_list = shap_values_rf[0].flatten().tolist()
            feat_values_list = X_scaled[0].flatten().tolist()
            shap_abs_list = shap_abs.flatten().tolist()

            top_indices = np.argsort(shap_abs)[-20:][::-1].tolist()

            top_features = []
            for i, idx in enumerate(top_indices):
                # Ensure idx is actually an integer
                if not isinstance(idx, int):
                    idx = int(idx) if hasattr(idx, '__int__') else idx[0]

                feature_name = f"feature_{idx}"
                try:
                    # Try to find actual feature name from importance list
                    for feat in self.feature_importance["top_20_features"]:
                        if feat["index"] == idx:
                            feature_name = feat["name"]
                            break
                except Exception:
                    pass

                top_features.append({
                    "feature": feature_name,
                    "feature_index": idx,
                    "shap_value": shap_values_list[idx],
                    "feature_value": feat_values_list[idx],
                    "importance": shap_abs_list[idx]
                })

            # Base value (expected model output) - handle different formats
            expected_val = explainer_rf.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                if len(expected_val) > 1:
                    base_value = float(expected_val[1])  # Positive class
                else:
                    base_value = float(expected_val[0])
            else:
                base_value = float(expected_val)

            return {
                "method": "SHAP TreeExplainer (Random Forest)",
                "base_value": base_value,
                "prediction": float(base_value + float(np.sum(shap_values_rf[0]))),
                "top_features": top_features,
                "interpretation": self._interpret_shap(top_features, base_value)
            }

        except Exception as e:
            import traceback
            return {"error": f"SHAP explanation failed: {str(e)}\n{traceback.format_exc()}"}

    def explain_lime(self, X):
        """
        Generate LIME explanations

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Dict with LIME values and interpretations
        """
        if not LIME_AVAILABLE:
            return {"error": "LIME not installed. Run: pip install lime"}

        try:
            X_scaled = self.scaler.transform(X)

            # Create wrapper for ensemble prediction
            def predict_fn(X_input):
                probas = []
                for x in X_input:
                    x_reshaped = x.reshape(1, -1)
                    proba = float(self._ensemble_predict_proba(x_reshaped)[0])
                    probas.append([1 - proba, proba])
                return np.array(probas)

            # Create LIME explainer
            explainer = LimeTabularExplainer(
                training_data=X_scaled,
                mode='classification',
                feature_names=[f"feature_{i}" for i in range(X_scaled.shape[1])],
                discretize_continuous=True
            )

            # Explain instance
            explanation = explainer.explain_instance(
                X_scaled[0],
                predict_fn,
                num_features=20,
                top_labels=1
            )

            # Get feature importances
            lime_values = explanation.as_list(label=1)

            top_features = []
            for i, (feature_desc, importance) in enumerate(lime_values[:20]):
                # Parse feature description (e.g., "feature_5 > 0.23")
                feature_name = feature_desc.split()[0]
                try:
                    feature_idx = int(feature_name.split('_')[1]) if '_' in feature_name else i
                except (ValueError, IndexError):
                    feature_idx = i

                # Get actual feature name if available
                actual_feature_name = feature_name
                try:
                    for feat in self.feature_importance["top_20_features"]:
                        if feat["index"] == feature_idx:
                            actual_feature_name = feat["name"]
                            break
                except Exception:
                    pass

                top_features.append({
                    "feature": actual_feature_name,
                    "feature_index": feature_idx,
                    "description": feature_desc,
                    "importance": float(importance),
                    "feature_value": float(X_scaled[0][feature_idx]) if feature_idx < len(X_scaled[0]) else 0.0
                })

            # Get prediction probability
            pred_proba = explanation.predict_proba[1] if hasattr(explanation, 'predict_proba') else 0.5

            return {
                "method": "LIME Tabular Explainer",
                "prediction_probability": float(pred_proba),
                "top_features": top_features,
                "interpretation": self._interpret_lime(top_features)
            }

        except Exception as e:
            import traceback
            return {"error": f"LIME explanation failed: {str(e)}\n{traceback.format_exc()}"}

    def visualize_feature_space(self, X):
        """
        Visualize feature space using PCA

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Dict with dimensionality reduction coordinates
        """
        X_scaled = self.scaler.transform(X)

        try:
            # For single sample, we can't do PCA but can show feature distribution
            if X_scaled.shape[0] == 1:
                # Create a pseudo-PCA using top variance features
                feature_variances = np.var(X_scaled, axis=1) if X_scaled.shape[0] > 1 else X_scaled[0] ** 2
                top_indices = np.argsort(feature_variances if len(feature_variances.shape) == 1 else feature_variances[0])[-2:][::-1].tolist()

                pc1_coord = float(X_scaled[0][top_indices[0]])
                pc2_coord = float(X_scaled[0][top_indices[1]] if len(top_indices) > 1 else 0.0)

                # Get feature names
                pc1_name = f"feature_{top_indices[0]}"
                pc2_name = f"feature_{top_indices[1]}" if len(top_indices) > 1 else "feature_0"

                try:
                    for feat in self.feature_importance["top_20_features"]:
                        if feat["index"] == top_indices[0]:
                            pc1_name = feat["name"]
                        if len(top_indices) > 1 and feat["index"] == top_indices[1]:
                            pc2_name = feat["name"]
                except Exception:
                    pass

                return {
                    "method": "Feature Projection (single sample)",
                    "coordinates": {
                        "pc1": pc1_coord,
                        "pc2": pc2_coord
                    },
                    "explained_variance": {
                        "pc1": 0.5,
                        "pc2": 0.5,
                        "total": 1.0
                    },
                    "principal_components": {
                        "pc1_top_features": [{"index": top_indices[0], "name": pc1_name, "weight": 1.0}],
                        "pc2_top_features": [{"index": top_indices[1] if len(top_indices) > 1 else 0, "name": pc2_name, "weight": 1.0}]
                    },
                    "interpretation": f"Single sample projection using top variance features: {pc1_name[:30]} and {pc2_name[:30]}"
                }

            # PCA (fast, linear) - for multiple samples
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(X_scaled)

            # Get explained variance
            explained_variance = pca.explained_variance_ratio_

            # Get top contributing features for each PC
            components = pca.components_
            pc1_features = []
            pc2_features = []

            for i, weight in enumerate(components[0]):
                # Get actual feature name if available
                feature_name = f"feature_{i}"
                try:
                    for feat in self.feature_importance["top_20_features"]:
                        if feat["index"] == i:
                            feature_name = feat["name"]
                            break
                except Exception:
                    pass

                pc1_features.append({"index": i, "name": feature_name, "weight": float(weight)})

            for i, weight in enumerate(components[1]):
                # Get actual feature name if available
                feature_name = f"feature_{i}"
                try:
                    for feat in self.feature_importance["top_20_features"]:
                        if feat["index"] == i:
                            feature_name = feat["name"]
                            break
                except Exception:
                    pass

                pc2_features.append({"index": i, "name": feature_name, "weight": float(weight)})

            # Sort by absolute weight
            pc1_features = sorted(pc1_features, key=lambda x: abs(x["weight"]), reverse=True)[:10]
            pc2_features = sorted(pc2_features, key=lambda x: abs(x["weight"]), reverse=True)[:10]

            pc1_coord = float(pca_coords[0][0])
            pc2_coord = float(pca_coords[0][1])
            total_var = float(sum(explained_variance))

            return {
                "method": "PCA (Principal Component Analysis)",
                "coordinates": {
                    "pc1": pc1_coord,
                    "pc2": pc2_coord
                },
                "explained_variance": {
                    "pc1": float(explained_variance[0]),
                    "pc2": float(explained_variance[1]),
                    "total": total_var
                },
                "principal_components": {
                    "pc1_top_features": pc1_features,
                    "pc2_top_features": pc2_features
                },
                "interpretation": f"This patient's EEG features map to coordinates ({pc1_coord:.2f}, {pc2_coord:.2f}) in the 2D PCA space, which captures {total_var * 100:.1f}% of the variance."
            }

        except Exception as e:
            import traceback
            return {"error": f"Feature space visualization failed: {str(e)}\n{traceback.format_exc()}"}

    def visualize_decision_path(self, X):
        """
        Visualize decision path through Random Forest

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Dict with decision path information
        """
        X_scaled = self.scaler.transform(X)

        try:
            # Get feature importances
            feature_importances = self.rf_model.feature_importances_
            top_feature_indices = np.argsort(feature_importances)[-20:][::-1].tolist()

            # Get one tree's path as example
            tree = self.rf_model.estimators_[0]
            tree_path = tree.decision_path(X_scaled).toarray()[0]
            node_indicator = tree_path.nonzero()[0].tolist()

            # Extract path information
            path_info = []
            for node_id in node_indicator:
                # Check if it's a leaf node
                if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:
                    class_dist = tree.tree_.value[node_id][0]
                    path_info.append({
                        "node_id": node_id,
                        "type": "leaf",
                        "class_distribution": [float(class_dist[0]), float(class_dist[1])],
                        "samples": int(tree.tree_.n_node_samples[node_id])
                    })
                else:
                    feature_idx = int(tree.tree_.feature[node_id])
                    threshold = float(tree.tree_.threshold[node_id])
                    feature_val = float(X_scaled[0][feature_idx])

                    # Get actual feature name
                    feature_name = f"feature_{feature_idx}"
                    try:
                        for feat in self.feature_importance["top_20_features"]:
                            if feat["index"] == feature_idx:
                                feature_name = feat["name"]
                                break
                    except Exception:
                        pass

                    path_info.append({
                        "node_id": node_id,
                        "type": "decision",
                        "feature_index": feature_idx,
                        "feature_name": feature_name,
                        "threshold": threshold,
                        "feature_value": feature_val,
                        "decision": "left" if feature_val <= threshold else "right",
                        "samples": int(tree.tree_.n_node_samples[node_id])
                    })

            # KNN neighbor analysis
            knn_distances, knn_indices = self.knn_model.kneighbors(X_scaled)

            # Build top features list with names
            top_features_list = []
            for idx in top_feature_indices:
                feature_name = f"feature_{idx}"
                try:
                    for feat in self.feature_importance["top_20_features"]:
                        if feat["index"] == idx:
                            feature_name = feat["name"]
                            break
                except Exception:
                    pass

                top_features_list.append({
                    "index": idx,
                    "name": feature_name,
                    "importance": float(feature_importances[idx])
                })

            return {
                "method": "Decision Path Analysis",
                "random_forest": {
                    "num_trees": len(self.rf_model.estimators_),
                    "example_tree_path": path_info,
                    "path_length": len(node_indicator),
                    "top_features": top_features_list
                },
                "knn_neighbors": {
                    "k": int(self.knn_model.n_neighbors),
                    "distances": [float(d) for d in knn_distances[0]],
                    "interpretation": f"The {self.knn_model.n_neighbors} nearest neighbors have distances: {[f'{d:.3f}' for d in knn_distances[0]]}"
                },
                "interpretation": f"The model's decision path went through {len(node_indicator)} nodes in the example tree. The {self.knn_model.n_neighbors} nearest neighbors influence the KNN prediction."
            }

        except Exception as e:
            import traceback
            return {"error": f"Decision path visualization failed: {str(e)}\n{traceback.format_exc()}"}

    def _interpret_shap(self, top_features, base_value):
        """Generate human-readable interpretation of SHAP values"""
        positive_features = [f for f in top_features if f["shap_value"] > 0]
        negative_features = [f for f in top_features if f["shap_value"] < 0]

        interpretation = f"Base prediction: {base_value:.3f}. "

        if positive_features:
            interpretation += f"Features increasing prediction: {', '.join([f['feature'] for f in positive_features[:3]])}. "

        if negative_features:
            interpretation += f"Features decreasing prediction: {', '.join([f['feature'] for f in negative_features[:3]])}."

        return interpretation

    def _interpret_lime(self, top_features):
        """Generate human-readable interpretation of LIME values"""
        positive_features = [f for f in top_features if f["importance"] > 0]
        negative_features = [f for f in top_features if f["importance"] < 0]

        interpretation = ""

        if positive_features:
            interpretation += f"Features supporting positive prediction: {', '.join([f['description'] for f in positive_features[:3]])}. "

        if negative_features:
            interpretation += f"Features supporting negative prediction: {', '.join([f['description'] for f in negative_features[:3]])}."

        return interpretation if interpretation else "No significant features identified."

    def visualize_tsne(self, X):
        """
        Visualize feature space using t-SNE (t-Distributed Stochastic Neighbor Embedding)

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Dict with t-SNE coordinates and training data context
        """
        X_scaled = self.scaler.transform(X)

        try:
            # For single sample, we need training data context
            if X_scaled.shape[0] == 1:
                # Load training data from main EEG CSV
                csv_path = Path(__file__).parent.parent / "EEG.machinelearing_data_BRMH.csv"

                try:
                    df = pd.read_csv(csv_path)

                    # Extract features (exclude metadata columns)
                    metadata_cols = ['no', 'no.', 'sex', 'age', 'eeg.date', 'education', 'IQ',
                                    'main.disorder', 'specific.disorder']
                    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
                    feature_cols = [col for col in df.columns if col not in metadata_cols + unnamed_cols]

                    # Get subset of training data (use 150 samples for better visualization)
                    train_samples = min(150, len(df))
                    X_train = df[feature_cols].head(train_samples).values.astype(float)
                    X_train_scaled = self.scaler.transform(X_train)

                    # Combine test sample with training data
                    X_combined = np.vstack([X_scaled, X_train_scaled])

                    # Calculate perplexity (must be less than n_samples)
                    total_samples = X_combined.shape[0]
                    perplexity = min(30, max(5, (total_samples - 1) // 3))

                    # Apply t-SNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
                    tsne_coords = tsne.fit_transform(X_combined)

                    # Extract test sample coordinates (first row)
                    test_coord_1 = float(tsne_coords[0][0])
                    test_coord_2 = float(tsne_coords[0][1])

                    # Get training coordinates for context
                    train_coords_1 = tsne_coords[1:, 0].tolist()
                    train_coords_2 = tsne_coords[1:, 1].tolist()

                    # Find nearest neighbors in t-SNE space
                    from scipy.spatial.distance import cdist
                    distances = cdist([tsne_coords[0]], tsne_coords[1:], metric='euclidean')[0]
                    nearest_indices = np.argsort(distances)[:5].tolist()
                    nearest_distances = [float(distances[i]) for i in nearest_indices]

                    return {
                        "method": "t-SNE (t-Distributed Stochastic Neighbor Embedding)",
                        "coordinates": {
                            "dim1": test_coord_1,
                            "dim2": test_coord_2
                        },
                        "training_context": {
                            "num_samples": train_samples,
                            "train_coords_1": train_coords_1,
                            "train_coords_2": train_coords_2,
                            "nearest_neighbors": {
                                "indices": nearest_indices,
                                "distances": nearest_distances
                            }
                        },
                        "interpretation": f"This patient's EEG features map to coordinates ({test_coord_1:.2f}, {test_coord_2:.2f}) in t-SNE space. t-SNE preserves local structure, showing which training samples have similar EEG patterns. The {len(nearest_indices)} nearest neighbors have distances: {[f'{d:.2f}' for d in nearest_distances[:3]]}."
                    }

                except Exception as e:
                    import traceback
                    error_msg = f"Failed to load training data: {str(e)}\n{traceback.format_exc()}"
                    return {"error": error_msg}

            # Multiple samples - direct t-SNE
            perplexity = min(30, max(5, (X_scaled.shape[0] - 1) // 3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            tsne_coords = tsne.fit_transform(X_scaled)

            return {
                "method": "t-SNE (t-Distributed Stochastic Neighbor Embedding)",
                "coordinates": {
                    "dim1": float(tsne_coords[0][0]),
                    "dim2": float(tsne_coords[0][1])
                },
                "all_coords_1": tsne_coords[:, 0].tolist(),
                "all_coords_2": tsne_coords[:, 1].tolist(),
                "interpretation": "t-SNE visualization showing local neighborhood structure in EEG feature space."
            }

        except Exception as e:
            import traceback
            return {"error": f"t-SNE visualization failed: {str(e)}\n{traceback.format_exc()}"}

    def visualize_umap(self, X):
        """
        Visualize feature space using UMAP (Uniform Manifold Approximation and Projection)

        Args:
            X: Feature array (n_samples, n_features)

        Returns:
            Dict with UMAP coordinates and training data context
        """
        if not UMAP_AVAILABLE:
            return {"error": "UMAP not available. Install with: pip install umap-learn"}

        X_scaled = self.scaler.transform(X)

        try:
            # For single sample, we need training data context
            if X_scaled.shape[0] == 1:
                # Load training data from main EEG CSV
                csv_path = Path(__file__).parent.parent / "EEG.machinelearing_data_BRMH.csv"

                try:
                    df = pd.read_csv(csv_path)

                    # Extract features (exclude metadata columns)
                    metadata_cols = ['no', 'no.', 'sex', 'age', 'eeg.date', 'education', 'IQ',
                                    'main.disorder', 'specific.disorder']
                    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
                    feature_cols = [col for col in df.columns if col not in metadata_cols + unnamed_cols]

                    # Get subset of training data (use 150 samples for better visualization)
                    train_samples = min(150, len(df))
                    X_train = df[feature_cols].head(train_samples).values.astype(float)
                    X_train_scaled = self.scaler.transform(X_train)

                    # Combine test sample with training data
                    X_combined = np.vstack([X_scaled, X_train_scaled])

                    # Calculate n_neighbors (must be >= 2 and < n_samples)
                    total_samples = X_combined.shape[0]
                    n_neighbors = min(15, max(2, total_samples - 1))

                    # Apply UMAP
                    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
                    umap_coords = reducer.fit_transform(X_combined)

                    # Extract test sample coordinates (first row)
                    test_coord_1 = float(umap_coords[0][0])
                    test_coord_2 = float(umap_coords[0][1])

                    # Get training coordinates for context
                    train_coords_1 = umap_coords[1:, 0].tolist()
                    train_coords_2 = umap_coords[1:, 1].tolist()

                    # Find nearest neighbors in UMAP space
                    from scipy.spatial.distance import cdist
                    distances = cdist([umap_coords[0]], umap_coords[1:], metric='euclidean')[0]
                    nearest_indices = np.argsort(distances)[:5].tolist()
                    nearest_distances = [float(distances[i]) for i in nearest_indices]

                    return {
                        "method": "UMAP (Uniform Manifold Approximation and Projection)",
                        "coordinates": {
                            "dim1": test_coord_1,
                            "dim2": test_coord_2
                        },
                        "training_context": {
                            "num_samples": train_samples,
                            "train_coords_1": train_coords_1,
                            "train_coords_2": train_coords_2,
                            "nearest_neighbors": {
                                "indices": nearest_indices,
                                "distances": nearest_distances
                            }
                        },
                        "interpretation": f"This patient's EEG features map to coordinates ({test_coord_1:.2f}, {test_coord_2:.2f}) in UMAP space. UMAP preserves both local and global structure, showing the overall data manifold. The {len(nearest_indices)} nearest neighbors have distances: {[f'{d:.2f}' for d in nearest_distances[:3]]}."
                    }

                except Exception as e:
                    import traceback
                    error_msg = f"Failed to load training data: {str(e)}\n{traceback.format_exc()}"
                    return {"error": error_msg}

            # Multiple samples - direct UMAP
            n_neighbors = min(15, max(2, X_scaled.shape[0] - 1))
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
            umap_coords = reducer.fit_transform(X_scaled)

            return {
                "method": "UMAP (Uniform Manifold Approximation and Projection)",
                "coordinates": {
                    "dim1": float(umap_coords[0][0]),
                    "dim2": float(umap_coords[0][1])
                },
                "all_coords_1": umap_coords[:, 0].tolist(),
                "all_coords_2": umap_coords[:, 1].tolist(),
                "interpretation": "UMAP visualization preserving both local and global structure in EEG feature space."
            }

        except Exception as e:
            import traceback
            return {"error": f"UMAP visualization failed: {str(e)}\n{traceback.format_exc()}"}
