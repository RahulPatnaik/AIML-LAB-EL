"""
Pytest Configuration and Shared Fixtures
"""

import pytest
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def models_dir():
    """Path to models directory"""
    return str(project_root / "all_models")


@pytest.fixture(scope="session")
def sample_csv_path():
    """Path to sample data CSV"""
    return str(project_root / "backend" / "sample_data.csv")


@pytest.fixture(scope="session")
def backend_url():
    """Backend API URL"""
    return "http://localhost:8000"


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require running API server"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers automatically
    for item in items:
        if "lime" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
