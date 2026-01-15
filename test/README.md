# Test Suite

Comprehensive tests for the EEG Explainability Dashboard.

## Test Structure

```
test/
├── test_api.py          # API endpoint tests (requires running backend)
├── test_explainer.py    # ModelExplainer unit tests (standalone)
├── conftest.py          # Pytest configuration and fixtures
└── README.md            # This file
```

## Installation

Install test dependencies:

```bash
pip install pytest requests
```

## Running Tests

### Run All Tests

```bash
cd /home/rahul/Desktop/AIML\ LAB\ EL
pytest test/ -v
```

### Run Specific Test Files

**Unit Tests (standalone, no server needed):**
```bash
pytest test/test_explainer.py -v
```

**API Tests (requires backend running):**
```bash
# Terminal 1 - Start backend
cd backend
python main.py

# Terminal 2 - Run API tests
pytest test/test_api.py -v
```

### Run Specific Tests

```bash
# Single test function
pytest test/test_explainer.py::TestModelExplainer::test_predict -v

# Single test class
pytest test/test_api.py::TestAPIEndpoints -v
```

### Skip Slow Tests

LIME tests can take 30-60 seconds. Skip them:

```bash
pytest test/ -v -m "not slow"
```

### Run Only API Tests

```bash
pytest test/ -v -m api
```

## Test Coverage

### API Tests (`test_api.py`)

Tests all 8 REST API endpoints:
- ✅ `GET /` - Health check
- ✅ `GET /disorders` - List available disorders
- ✅ `GET /sample` - Get sample patient data
- ✅ `POST /predict` - Make predictions
- ✅ `POST /explain/shap` - SHAP explanations
- ✅ `POST /explain/lime` - LIME explanations
- ✅ `POST /visualize/feature_space` - PCA visualization
- ✅ `POST /visualize/decision_path` - Decision path analysis

**Performance tests:**
- Prediction speed (<2 seconds)
- SHAP speed (<5 seconds)

### Explainer Tests (`test_explainer.py`)

Tests the `ModelExplainer` class directly:
- ✅ Initialization and model loading
- ✅ Prediction logic
- ✅ Ensemble prediction combining
- ✅ SHAP explanations
- ✅ LIME explanations
- ✅ PCA feature space visualization
- ✅ Decision path visualization
- ✅ Feature name resolution
- ✅ Confidence level calculation
- ✅ Error handling

## Expected Test Results

**All passing:**
```
test_api.py::TestAPIEndpoints::test_health_check PASSED
test_api.py::TestAPIEndpoints::test_get_disorders PASSED
test_api.py::TestAPIEndpoints::test_predict_endpoint PASSED
test_api.py::TestAPIEndpoints::test_shap_explanation PASSED
...
test_explainer.py::TestModelExplainer::test_predict PASSED
test_explainer.py::TestModelExplainer::test_explain_shap PASSED
...
==================== X passed in Y.YYs ====================
```

## Debugging Failed Tests

### Common Issues

**1. API tests fail with connection error:**
```
requests.exceptions.ConnectionError: Connection refused
```
**Solution:** Start the backend server first:
```bash
cd backend && python main.py
```

**2. SHAP/LIME tests skipped:**
```
SKIPPED [1] test_explainer.py:XX: SHAP not available
```
**Solution:** Install SHAP and LIME:
```bash
pip install shap lime
```

**3. Module import errors:**
```
ModuleNotFoundError: No module named 'explainer'
```
**Solution:** The `conftest.py` handles path setup. Run from project root:
```bash
cd /home/rahul/Desktop/AIML\ LAB\ EL
pytest test/
```

## Writing New Tests

### Example: Test a new API endpoint

```python
# In test_api.py
def test_new_endpoint(self):
    """Test /new/endpoint"""
    payload = {"param": "value"}
    response = requests.post(f"{BASE_URL}/new/endpoint", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "expected_key" in data
```

### Example: Test a new explainer method

```python
# In test_explainer.py
def test_new_method(self):
    """Test new explainer method"""
    result = self.explainer.new_method(self.sample_features)

    assert isinstance(result, dict)
    assert "expected_field" in result
```

## CI/CD Integration

To run tests in GitHub Actions:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r backend/requirements.txt
      - run: pip install pytest requests
      - run: pytest test/test_explainer.py -v
      - run: |
          cd backend && python main.py &
          sleep 5
          cd ..
          pytest test/test_api.py -v
```

## Performance Benchmarks

Expected execution times:
- **Unit tests:** ~5-10 seconds
- **API tests (without LIME):** ~15-20 seconds
- **LIME tests:** ~30-60 seconds each
- **Full suite:** ~2-3 minutes

## Test Data

Tests use:
- Sample patient data from `backend/sample_data.csv`
- Models from `all_models/` directory (300+ MB)
- First available disorder from deployment manifest

## Advanced Usage

### Run with coverage

```bash
pip install pytest-cov
pytest test/ --cov=backend --cov-report=html
# Open htmlcov/index.html
```

### Run with detailed output

```bash
pytest test/ -vv --tb=long
```

### Run in parallel (faster)

```bash
pip install pytest-xdist
pytest test/ -n auto
```

### Generate XML report (for CI)

```bash
pytest test/ --junitxml=test-results.xml
```

## Troubleshooting

### Tests hang or timeout

**LIME tests can be very slow.** Use:
```bash
pytest test/ -m "not slow"  # Skip LIME tests
```

Or increase timeout:
```bash
pytest test/ --timeout=300  # 5 minute timeout
```

### Memory issues

Large model files (~300MB) may cause issues on constrained systems. Close other applications before running tests.

### Port already in use

If backend fails to start on port 8000:
```bash
lsof -ti:8000 | xargs kill -9  # Kill process on port 8000
```

## Contact

For issues with tests, check:
1. Backend is running (for API tests)
2. All dependencies installed (`pip install -r backend/requirements.txt`)
3. Python 3.8+ is used
4. Models directory exists and is populated

Happy testing! 🧪
