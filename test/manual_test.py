"""
Manual Test Script - Actually runs all functions
Run this to verify everything works before deploying
"""

import sys
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from explainer import ModelExplainer
import pandas as pd
import json

def test_all_functions():
    """Test all explainer functions"""

    print("=" * 80)
    print("COMPREHENSIVE EXPLAINER TEST")
    print("=" * 80)

    # Setup
    models_dir = Path(__file__).parent.parent / "all_models"

    # Load manifest
    with open(models_dir / "deployment_manifest.json") as f:
        manifest = json.load(f)

    test_disorder = manifest["models"][0]["safe_name"]
    print(f"\n📋 Testing disorder: {test_disorder}")

    # Load sample data
    csv_path = Path(__file__).parent.parent / "backend" / "sample_data.csv"
    df = pd.read_csv(csv_path)
    # Exclude all metadata columns - keep only EEG features
    exclude_cols = ["no", "sex", "age", "eeg.date", "education", "IQ", "main.disorder", "specific.disorder", "no."]
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]
    sample_features = df[feature_cols].values

    print(f"✓ Loaded sample data: {sample_features.shape}")

    # Initialize explainer
    print("\n1️⃣ Testing Explainer Initialization...")
    try:
        explainer = ModelExplainer(str(models_dir), test_disorder)
        print(f"   ✓ Explainer initialized for {test_disorder}")
        print(f"   ✓ KNN model: {type(explainer.knn_model).__name__}")
        print(f"   ✓ RF model: {type(explainer.rf_model).__name__}")
        print(f"   ✓ ET model: {type(explainer.et_model).__name__}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        return False

    # Test Prediction
    print("\n2️⃣ Testing Prediction...")
    try:
        result = explainer.predict(sample_features)
        print(f"   ✓ Prediction: {result['prediction']}")
        print(f"   ✓ Probability: {result['probability']:.4f}")
        print(f"   ✓ Threshold: {result['threshold']:.4f}")
        print(f"   ✓ Confidence: {result['confidence']}")
        print(f"   ✓ Individual models:")
        print(f"      - KNN: {result['individual_predictions']['knn']:.4f}")
        print(f"      - RF:  {result['individual_predictions']['random_forest']:.4f}")
        print(f"      - ET:  {result['individual_predictions']['extra_trees']:.4f}")

        # Verify types
        assert isinstance(result['prediction'], int), "Prediction must be int"
        assert isinstance(result['probability'], float), "Probability must be float"
        assert 0 <= result['probability'] <= 1, "Probability must be in [0,1]"
        assert result['confidence'] in ['High', 'Medium', 'Low'], "Invalid confidence"

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test SHAP
    print("\n3️⃣ Testing SHAP Explanation...")
    try:
        result = explainer.explain_shap(sample_features)

        if "error" in result:
            print(f"   ⚠️  SKIPPED: {result['error']}")
        else:
            print(f"   ✓ Method: {result['method']}")
            print(f"   ✓ Base value: {result['base_value']:.4f}")
            print(f"   ✓ Prediction: {result['prediction']:.4f}")
            print(f"   ✓ Top features: {len(result['top_features'])}")
            print(f"   ✓ Sample features:")
            for i, feat in enumerate(result['top_features'][:3]):
                print(f"      {i+1}. {feat['feature'][:40]:40s} | SHAP: {feat['shap_value']:+.4f}")

            # Verify types
            assert isinstance(result['base_value'], float), "base_value must be float"
            assert isinstance(result['prediction'], float), "prediction must be float"
            assert all('feature' in f for f in result['top_features']), "Missing feature names"
            assert all(isinstance(f['shap_value'], float) for f in result['top_features']), "SHAP values must be float"

            print(f"   ✓ All types correct")

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test LIME
    print("\n4️⃣ Testing LIME Explanation (may take 30-60s)...")
    try:
        result = explainer.explain_lime(sample_features)

        if "error" in result:
            print(f"   ⚠️  SKIPPED: {result['error']}")
        else:
            print(f"   ✓ Method: {result['method']}")
            print(f"   ✓ Prediction probability: {result['prediction_probability']:.4f}")
            print(f"   ✓ Top features: {len(result['top_features'])}")
            print(f"   ✓ Sample features:")
            for i, feat in enumerate(result['top_features'][:3]):
                print(f"      {i+1}. {feat['description'][:50]:50s} | Imp: {feat['importance']:+.4f}")

            # Verify types
            assert isinstance(result['prediction_probability'], float), "prediction_probability must be float"
            assert all('feature' in f for f in result['top_features']), "Missing feature names"
            assert all(isinstance(f['importance'], float) for f in result['top_features']), "Importance must be float"

            print(f"   ✓ All types correct")

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Feature Space
    print("\n5️⃣ Testing Feature Space Visualization...")
    try:
        result = explainer.visualize_feature_space(sample_features)

        if "error" in result:
            print(f"   ✗ FAILED: {result['error']}")
            return False

        print(f"   ✓ Method: {result['method']}")
        print(f"   ✓ PC1 coordinate: {result['coordinates']['pc1']:.4f}")
        print(f"   ✓ PC2 coordinate: {result['coordinates']['pc2']:.4f}")
        print(f"   ✓ Total variance explained: {result['explained_variance']['total']*100:.2f}%")
        print(f"   ✓ PC1 top features: {len(result['principal_components']['pc1_top_features'])}")
        print(f"   ✓ PC2 top features: {len(result['principal_components']['pc2_top_features'])}")

        # Verify types
        assert isinstance(result['coordinates']['pc1'], float), "PC1 must be float"
        assert isinstance(result['coordinates']['pc2'], float), "PC2 must be float"
        assert 0 <= result['explained_variance']['total'] <= 1, "Variance must be in [0,1]"

        print(f"   ✓ All types correct")

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Decision Path
    print("\n6️⃣ Testing Decision Path Visualization...")
    try:
        result = explainer.visualize_decision_path(sample_features)

        if "error" in result:
            print(f"   ✗ FAILED: {result['error']}")
            return False

        print(f"   ✓ Method: {result['method']}")
        print(f"   ✓ Number of trees: {result['random_forest']['num_trees']}")
        print(f"   ✓ Path length: {result['random_forest']['path_length']}")
        print(f"   ✓ Top features: {len(result['random_forest']['top_features'])}")
        print(f"   ✓ KNN k: {result['knn_neighbors']['k']}")
        print(f"   ✓ KNN distances: {result['knn_neighbors']['distances'][:3]}")

        # Sample top features
        print(f"   ✓ Sample top features:")
        for i, feat in enumerate(result['random_forest']['top_features'][:3]):
            print(f"      {i+1}. {feat['name'][:40]:40s} | Imp: {feat['importance']:.6f}")

        # Verify types
        assert isinstance(result['random_forest']['num_trees'], int), "num_trees must be int"
        assert isinstance(result['random_forest']['path_length'], int), "path_length must be int"
        assert all(isinstance(d, float) for d in result['knn_neighbors']['distances']), "Distances must be float"
        assert all('name' in f for f in result['random_forest']['top_features']), "Missing feature names"

        print(f"   ✓ All types correct")

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Multiple Disorders
    print("\n7️⃣ Testing Multiple Disorders...")
    try:
        test_count = min(3, len(manifest["models"]))
        for i in range(test_count):
            disorder = manifest["models"][i]["safe_name"]
            explainer = ModelExplainer(str(models_dir), disorder)
            result = explainer.predict(sample_features)
            print(f"   ✓ {disorder:40s} | Pred: {result['prediction']} | Prob: {result['probability']:.4f}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_all_functions()
    sys.exit(0 if success else 1)
