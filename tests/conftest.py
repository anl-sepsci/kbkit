"""
Pytest configuration and shared fixtures for kbkit tests.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Generator:
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix="kbkit_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'name': ['compound_A', 'compound_B', 'compound_C'],
        'value': [1.5, 2.3, 3.7],
        'category': ['type1', 'type2', 'type1'],
        'timestamp': pd.date_range('2024-01-01', periods=3)
    })


@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Create a sample numpy array for testing."""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create a sample configuration dictionary."""
    return {
        'name': 'test_property',
        'units': 'kg/mol',
        'avg': True,
        'precision': 3,
        'cache_enabled': True
    }


@pytest.fixture
def mock_cache():
    """Create a mock cache for testing decorators."""
    class MockCache:
        def __init__(self):
            self._cache = {}

        def get(self, key, default=None):
            return self._cache.get(key, default)

        def set(self, key, value):
            self._cache[key] = value

        def clear(self):
            self._cache.clear()

        def __contains__(self, key):
            return key in self._cache

    return MockCache()


@pytest.fixture
def temp_file(test_data_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    file_path = test_data_dir / "temp_test_file.txt"
    file_path.touch()
    yield file_path
    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def sample_json_data() -> Dict[str, Any]:
    """Create sample JSON data for testing."""
    return {
        'metadata': {
            'version': '1.0',
            'created': '2024-01-01',
            'author': 'test_user'
        },
        'data': [
            {'id': 1, 'value': 10.5},
            {'id': 2, 'value': 20.3},
            {'id': 3, 'value': 15.7}
        ]
    }


@pytest.fixture
def sample_csv_content() -> str:
    """Create sample CSV content for testing."""
    return """name,value,category
compound_A,1.5,type1
compound_B,2.3,type2
compound_C,3.7,type1"""


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    import os
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)

import warnings

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def configure_test_warnings():
    """Configure warning filters for tests."""
    # For NumPy < 2.0
    try:
        warnings.filterwarnings('ignore', category=np.RankWarning)
    except AttributeError:
        # NumPy 2.0+ doesn't have RankWarning, use generic approach
        warnings.filterwarnings('ignore', message="Polyfit may be poorly conditioned")

    # Other warnings
    warnings.filterwarnings('ignore', category=UserWarning,
                        message=".*input contained no data.*")
    warnings.filterwarnings('ignore', category=RuntimeWarning,
                        message="Mean of empty slice")


