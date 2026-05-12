"""pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable in tests without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def quadratic_params() -> np.ndarray:
    """Starting parameter vector for optimizer tests."""
    return np.array([3.0, -2.0])


@pytest.fixture
def simple_grads() -> np.ndarray:
    """Simple gradient vector for single-step math verification."""
    return np.array([1.0, -0.5])
