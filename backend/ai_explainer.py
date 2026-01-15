"""
AI-Powered Clinical Explanation Module
Uses Mistral API to generate natural language explanations
"""

import os
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Initialize Mistral client
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None


def generate_prediction_explanation(disorder, prediction_data, patient_metadata=None):
    """Generate AI explanation for prediction results"""

    if not client:
        return {
            "explanation": "AI explanations unavailable (Mistral API key not configured)",
            "error": True
        }

    try:
        # Extract correct values
        prediction_label = "PRESENT" if prediction_data.get('prediction', 0) == 1 else "ABSENT"
        probability = prediction_data.get('probability', 0)
        confidence = prediction_data.get('confidence', 'Unknown')
        threshold = prediction_data.get('threshold', 0.5)

        individual = prediction_data.get('individual_predictions', {})
        knn_prob = individual.get('knn', 0)
        rf_prob = individual.get('random_forest', 0)
        et_prob = individual.get('extra_trees', 0)

        prompt = f"""You are a clinical AI assistant explaining psychiatric EEG screening results to healthcare providers.

Disorder: {disorder.replace('_', ' ')}
Patient: {patient_metadata if patient_metadata else 'Sample patient, Male, 57 years old'}

Prediction Results:
- Ensemble Prediction: {prediction_label}
- Probability: {probability:.2%} ({probability:.4f})
- Confidence: {confidence}
- Optimal Threshold: {threshold:.2%} ({threshold:.4f})

Individual Model Probabilities:
- KNN: {knn_prob:.2%} ({knn_prob:.4f})
- Random Forest: {rf_prob:.2%} ({rf_prob:.4f})
- Extra Trees: {et_prob:.2%} ({et_prob:.4f})

Task: Provide a 3-4 sentence clinical interpretation explaining:
1. What this prediction means for the patient (use the ACTUAL probability values above)
2. Why the confidence level is {confidence} (consider distance from threshold)
3. How the three models agree or disagree (compare the individual probabilities)
4. Recommended next steps for clinical evaluation

IMPORTANT: Use the EXACT probability values provided above in your explanation. Do not make up different numbers.
Keep language professional but accessible. Focus on clinical actionability.
Do NOT use markdown formatting (no asterisks, no bold, no italic) - use plain text only."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

        return {
            "explanation": explanation,
            "error": False
        }

    except Exception as e:
        return {
            "explanation": f"AI explanation generation failed: {str(e)}",
            "error": True
        }


def generate_shap_explanation(disorder, shap_data, patient_metadata=None):
    """Generate AI explanation for SHAP results"""

    if not client:
        return {
            "explanation": "AI explanations unavailable (Mistral API key not configured)",
            "error": True
        }

    try:
        # Extract top features
        top_features_text = "\n".join([
            f"- {f['feature']}: SHAP value {f['shap_value']:.3f}, feature value {f['feature_value']:.3f}"
            for f in shap_data.get('top_features', [])[:5]
        ])

        prompt = f"""You are a clinical AI assistant explaining SHAP feature attributions for psychiatric EEG analysis.

Disorder: {disorder.replace('_', ' ')}
Patient: {patient_metadata if patient_metadata else 'Sample patient, Male, 57 years old'}

SHAP Analysis Results:
Base Value (average prediction): {shap_data.get('base_value', 0):.3f}

Top 5 Contributing Features:
{top_features_text}

Task: Provide a 3-4 sentence clinical explanation covering:
1. What SHAP values mean (how features push prediction up or down from baseline)
2. Clinical significance of the top contributing features (relate to known EEG biomarkers)
3. How these patterns relate to typical {disorder.replace('_', ' ')} presentations
4. What brain regions/frequency bands are most relevant

Use clinical terminology but keep it accessible. Connect EEG features to neurophysiology.
Do NOT use markdown formatting (no asterisks, no bold, no italic) - use plain text only."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

        return {
            "explanation": explanation,
            "error": False
        }

    except Exception as e:
        return {
            "explanation": f"AI explanation generation failed: {str(e)}",
            "error": True
        }


def generate_tsne_explanation(disorder, tsne_data, patient_metadata=None):
    """Generate AI explanation for t-SNE visualization"""

    if not client:
        return {
            "explanation": "AI explanations unavailable (Mistral API key not configured)",
            "error": True
        }

    try:
        nearest_neighbors = tsne_data.get('training_context', {}).get('nearest_neighbors', {})
        distances = nearest_neighbors.get('distances', [])

        prompt = f"""You are a clinical AI assistant explaining t-SNE latent space visualizations for psychiatric EEG.

Disorder: {disorder.replace('_', ' ')}
Patient: {patient_metadata if patient_metadata else 'Sample patient, Male, 57 years old'}

t-SNE Analysis:
- Patient coordinates: ({tsne_data.get('coordinates', {}).get('dim1', 0):.2f}, {tsne_data.get('coordinates', {}).get('dim2', 0):.2f})
- Training samples shown: {tsne_data.get('training_context', {}).get('num_samples', 0)}
- Nearest neighbor distances: {', '.join([f'{d:.2f}' for d in distances[:3]])}

Context: t-SNE is a non-linear dimensionality reduction technique that preserves local neighborhood structure. Patients with similar EEG patterns cluster together in 2D space.

Task: Provide a 3-4 sentence clinical explanation covering:
1. What it means that this patient is at these coordinates in t-SNE space
2. Interpretation of the nearest neighbor distances (small = very similar, large = unusual pattern)
3. What clustering with other patients suggests about their EEG pattern
4. Clinical implications for {disorder.replace('_', ' ')} screening

Focus on what the visualization tells us about this patient's brain activity patterns.
Do NOT use markdown formatting (no asterisks, no bold, no italic) - use plain text only."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

        return {
            "explanation": explanation,
            "error": False
        }

    except Exception as e:
        return {
            "explanation": f"AI explanation generation failed: {str(e)}",
            "error": True
        }


def generate_umap_explanation(disorder, umap_data, patient_metadata=None):
    """Generate AI explanation for UMAP visualization"""

    if not client:
        return {
            "explanation": "AI explanations unavailable (Mistral API key not configured)",
            "error": True
        }

    try:
        nearest_neighbors = umap_data.get('training_context', {}).get('nearest_neighbors', {})
        distances = nearest_neighbors.get('distances', [])

        prompt = f"""You are a clinical AI assistant explaining UMAP manifold visualizations for psychiatric EEG.

Disorder: {disorder.replace('_', ' ')}
Patient: {patient_metadata if patient_metadata else 'Sample patient, Male, 57 years old'}

UMAP Analysis:
- Patient coordinates: ({umap_data.get('coordinates', {}).get('dim1', 0):.2f}, {umap_data.get('coordinates', {}).get('dim2', 0):.2f})
- Training samples shown: {umap_data.get('training_context', {}).get('num_samples', 0)}
- Nearest neighbor distances: {', '.join([f'{d:.2f}' for d in distances[:3]])}

Context: UMAP preserves both LOCAL structure (similar patients stay close) and GLOBAL structure (overall data topology). It reveals the "manifold" of psychiatric EEG patterns.

Task: Provide a 3-4 sentence clinical explanation covering:
1. What UMAP reveals about this patient's position in the overall EEG landscape
2. How the global structure helps identify disorder subtypes and comorbidities
3. Interpretation of where the patient lies (central cluster vs outlier vs boundary region)
4. What this suggests for {disorder.replace('_', ' ')} screening confidence

Emphasize UMAP's advantage over t-SNE in showing the "big picture" of disorder relationships.
Do NOT use markdown formatting (no asterisks, no bold, no italic) - use plain text only."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

        return {
            "explanation": explanation,
            "error": False
        }

    except Exception as e:
        return {
            "explanation": f"AI explanation generation failed: {str(e)}",
            "error": True
        }


def generate_pca_explanation(disorder, pca_data, patient_metadata=None):
    """Generate AI explanation for PCA visualization"""

    if not client:
        return {
            "explanation": "AI explanations unavailable (Mistral API key not configured)",
            "error": True
        }

    try:
        explained_var = pca_data.get('explained_variance', {})

        prompt = f"""You are a clinical AI assistant explaining PCA feature space for psychiatric EEG.

Disorder: {disorder.replace('_', ' ')}
Patient: {patient_metadata if patient_metadata else 'Sample patient, Male, 57 years old'}

PCA Analysis:
- Patient coordinates: PC1={pca_data.get('coordinates', {}).get('pc1', 0):.2f}, PC2={pca_data.get('coordinates', {}).get('pc2', 0):.2f}
- PC1 explains {explained_var.get('pc1', 0)*100:.1f}% variance
- PC2 explains {explained_var.get('pc2', 0)*100:.1f}% variance
- Total variance captured: {explained_var.get('total', 0)*100:.1f}%

Context: PCA is a linear projection showing maximum variance directions. PC1 and PC2 represent the most important combinations of EEG features (typically dominated by delta/theta frequency bands).

Task: Provide a 3-4 sentence clinical explanation covering:
1. What the PC1 and PC2 coordinates reveal about this patient's EEG
2. Which frequency bands dominate these principal components (usually delta/theta)
3. How this patient's position compares to typical {disorder.replace('_', ' ')} patterns
4. Clinical interpretation of the explained variance percentage

Keep focus on what the linear projection tells us about dominant EEG characteristics.
Do NOT use markdown formatting (no asterisks, no bold, no italic) - use plain text only."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

        return {
            "explanation": explanation,
            "error": False
        }

    except Exception as e:
        return {
            "explanation": f"AI explanation generation failed: {str(e)}",
            "error": True
        }


def generate_lime_explanation(disorder, lime_data, patient_metadata=None):
    """Generate AI explanation for LIME results"""

    if not client:
        return {
            "explanation": "AI explanations unavailable (Mistral API key not configured)",
            "error": True
        }

    try:
        # Extract top features
        top_features_text = "\n".join([
            f"- {f['description']}: importance {f['importance']:.3f}"
            for f in lime_data.get('top_features', [])[:5]
        ])

        prompt = f"""You are a clinical AI assistant explaining LIME local explanations for psychiatric EEG.

Disorder: {disorder.replace('_', ' ')}
Patient: {patient_metadata if patient_metadata else 'Sample patient, Male, 57 years old'}

LIME Analysis Results:
Top 5 Feature Rules:
{top_features_text}

Context: LIME creates a simple linear model around this specific patient, showing which features would push the prediction toward "Present" (positive importance) or "Absent" (negative importance) for {disorder.replace('_', ' ')}.

Task: Provide a 3-4 sentence clinical explanation covering:
1. What LIME rules reveal about this patient's specific EEG pattern
2. How to interpret positive vs negative importance scores
3. Which rules align with known biomarkers for {disorder.replace('_', ' ')}
4. How LIME complements SHAP (rules vs attributions)

Emphasize that LIME provides actionable "if-then" rules for clinicians.
Do NOT use markdown formatting (no asterisks, no bold, no italic) - use plain text only."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

        return {
            "explanation": explanation,
            "error": False
        }

    except Exception as e:
        return {
            "explanation": f"AI explanation generation failed: {str(e)}",
            "error": True
        }


def generate_decision_path_explanation(disorder, decision_data, patient_metadata=None):
    """Generate AI explanation for decision path analysis"""

    if not client:
        return {
            "explanation": "AI explanations unavailable (Mistral API key not configured)",
            "error": True
        }

    try:
        rf_data = decision_data.get('random_forest', {})
        knn_data = decision_data.get('knn_neighbors', {})

        prompt = f"""You are a clinical AI assistant explaining decision path analysis for psychiatric EEG ensemble models.

Disorder: {disorder.replace('_', ' ')}
Patient: {patient_metadata if patient_metadata else 'Sample patient, Male, 57 years old'}

Decision Path Analysis:
Random Forest:
- Number of trees: {rf_data.get('num_trees', 0)}
- Example tree path length: {rf_data.get('path_length', 0)} nodes
- Top features used in trees: {len(rf_data.get('top_features', []))}

KNN:
- K neighbors: {knn_data.get('k', 0)}
- Neighbor distances: {', '.join([f"{d:.3f}" for d in knn_data.get('distances', [])[:3]])}

Context: The ensemble uses Random Forest (feature interactions through decision trees) and KNN (local similarity). This shows HOW the models made their prediction.

Task: Provide a 3-4 sentence clinical explanation covering:
1. What the decision tree path reveals about feature-based reasoning
2. Interpretation of KNN neighbor distances (close neighbors = high confidence)
3. How RF and KNN provide complementary perspectives (global rules vs local similarity)
4. What this tells us about confidence in the {disorder.replace('_', ' ')} prediction

Focus on transparency - showing clinicians the model's internal reasoning process.
Do NOT use markdown formatting (no asterisks, no bold, no italic) - use plain text only."""

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content

        return {
            "explanation": explanation,
            "error": False
        }

    except Exception as e:
        return {
            "explanation": f"AI explanation generation failed: {str(e)}",
            "error": True
        }


def generate_full_check_explanation(all_predictions, patient_metadata):
    """Generate AI explanation for full disorder check"""
    
    if not client:
        return {
            "explanation": "AI explanations unavailable (Mistral API key not configured)",
            "error": True
        }
    
    try:
        # Format all predictions
        predictions_text = []
        for result in all_predictions:
            disorder = result['disorder'].replace('_', ' ')
            prob = result['probability']
            pred = "PRESENT" if result['prediction'] == 1 else "ABSENT"
            conf = result['confidence']
            predictions_text.append(f"- {disorder}: {pred} ({prob:.1%} probability, {conf} confidence)")
        
        predictions_summary = "\n".join(predictions_text)
        
        prompt = f"""You are a clinical AI assistant analyzing comprehensive psychiatric screening results from EEG brainwave data.

Patient: {patient_metadata}

FULL DISORDER SCREENING RESULTS:
{predictions_summary}

Based on these results from 17 different psychiatric disorder models:

1. Which disorder(s) appear most likely based on the probabilities and confidence levels?
2. Are there any co-occurring disorders (comorbidities) indicated by multiple positive predictions?
3. What patterns do you notice in the predictions? (e.g., mood-related, anxiety-related, etc.)
4. What should be the next steps for clinical evaluation?

Provide a concise but comprehensive clinical interpretation in 3-4 paragraphs.

IMPORTANT: Do NOT use markdown formatting (no asterisks, no bold, no italic) - use plain text only."""

        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "explanation": response.choices[0].message.content.strip(),
            "error": False
        }
        
    except Exception as e:
        return {
            "explanation": f"Error generating AI explanation: {str(e)}",
            "error": True
        }
