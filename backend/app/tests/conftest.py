import os, sys, pathlib
import pytest

# Ensure /app (WORKDIR) is on the path so `from app.main import app` works
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# Set env vars before importing the app so settings picks them up
os.environ.setdefault("VECTOR_STORE", "memory")
os.environ.setdefault("LLM_PROVIDER", "stub")
# When running locally (outside Docker), point at the real data dir
_repo_root = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("DATA_DIR", str(_repo_root / "data"))

from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client():
    return TestClient(app)
