from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import create_app


@pytest.fixture(scope='session')
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def payload():
    return {'limit': 50000, 'age': 35, 'sex': 1, 'edu': 2, 'mar': 2,
            'pay': [0] * 6, 'bill': [20000] * 6, 'paid': [2000] * 6}
