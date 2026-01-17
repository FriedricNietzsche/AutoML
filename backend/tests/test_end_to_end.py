"""
End-to-end integration tests (networked: Kaggle + OpenRouter).
Requires env:
  OPENROUTER_API_KEY, KAGGLE_USERNAME, KAGGLE_KEY
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append("backend")
from app.main import app  # noqa: E402


HAS_CREDS = bool(os.getenv("OPENROUTER_API_KEY")) and bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
client = TestClient(app) if HAS_CREDS else None


def run_tabular_flow(project_id: str, dataset_ref: str, target: str):
    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "Build a predictor that predicts house prices"})
    assert r1.status_code == 200

    r2 = client.post(f"/api/projects/{project_id}/ingest/kaggle", json={"dataset": dataset_ref})
    assert r2.status_code == 200

    r3 = client.post(f"/api/projects/{project_id}/train/tabular", json={"target": target, "task_type": "classification"})
    assert r3.status_code == 200
    metrics = r3.json().get("metrics", {})

    r4 = client.post(f"/api/projects/{project_id}/report", json={"profile": r2.json(), "metrics": metrics})
    assert r4.status_code == 200

    r5 = client.post(f"/api/projects/{project_id}/export")
    assert r5.status_code == 200

    return {"metrics": metrics, "artifacts": r3.json().get("artifacts", {}), "export": r5.json().get("export")}


@pytest.mark.skipif(not HAS_CREDS, reason="Requires OPENROUTER_API_KEY and Kaggle creds")
def test_tabular_iris_flow():
    result = run_tabular_flow("e2e-iris", "uciml/iris", "Species")
    assert "metrics" in result
    assert result["metrics"].get("accuracy") is not None


def run_image_flow(project_id: str, dataset_ref: str, data_subdir: str):
    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "Build a cat vs dog classifier"})
    assert r1.status_code == 200

    r2 = client.post(f"/api/projects/{project_id}/ingest/kaggle", json={"dataset": dataset_ref})
    assert r2.status_code == 200

    # Expect images under data/assets/projects/{id}/images after manual unzip; here we just trigger trainer
    r3 = client.post(f"/api/projects/{project_id}/train/image", json={"data_subdir": data_subdir})
    assert r3.status_code == 200
    return r3.json()


@pytest.mark.skipif(not HAS_CREDS, reason="Requires OPENROUTER_API_KEY and Kaggle creds; needs images extracted")
def test_image_flow_catsdogs():
    pytest.skip("Image flow requires a pre-extracted public cats/dogs dataset; skip in CI")
