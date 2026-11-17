import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nutrimate.data import load_recipes  # noqa: E402


@pytest.fixture(scope="session")
def recipes():
    return load_recipes()


@pytest.fixture(scope="session")
def recommender(recipes):
    from nutrimate.hybrid import Recommender
    return Recommender.load()
