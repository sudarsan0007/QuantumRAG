import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def engine():
    from quantumrag.engine import QuantumRAG
    return QuantumRAG(max_docs=300)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
