"""
End-to-end flow for handwritten digit classification (MNIST via Hugging Face).
Project id matches filename: e2e-digits-demo.
"""
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sys
sys.path.append("backend")
from app.main import app  # noqa: E402


HAS_CREDS = bool(os.getenv("OPENROUTER_API_KEY"))
client = TestClient(app) if HAS_CREDS else None


@pytest.mark.skipif(not HAS_CREDS, reason="Requires OPENROUTER_API_KEY")
def test_full_digits_classification():
    project_id = "e2e-digits-demo"
    project_dir = Path("data/assets/projects") / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir)

    # 1) Parse intent
    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "Build a classifier for handwritten digits"})
    assert r1.status_code == 200

    # 2) Ingest using HF MNIST (small sample for speed)
    hf_dataset = "mnist"
    r2 = client.post(
        f"/api/projects/{project_id}/ingest/hf",
        json={"dataset": hf_dataset, "split": "train", "max_rows": 50},
    )
    assert r2.status_code == 200
    assert "label" in [c.lower() for c in r2.json().get("columns", [])]

    # 3) Train classification
    r3 = client.post(
        f"/api/projects/{project_id}/train/tabular",
        json={"target": "label", "task_type": "classification", "model_id": "logreg"},
    )
    assert r3.status_code == 200
    metrics = r3.json().get("metrics", {})
    artifacts = r3.json().get("artifacts", {})
    assert "accuracy" in metrics
    assert "confusion" in artifacts

    # 4) Report with source metadata for notebook
    context = {
        "profile": r2.json(),
        "metrics": metrics,
        "task_type": "classification",
        "target": "label",
        "source": {"hf_dataset": hf_dataset, "split": "train"},
    }
    r4 = client.post(f"/api/projects/{project_id}/report", json=context)
    assert r4.status_code == 200

    # 5) Export
    r5 = client.post(f"/api/projects/{project_id}/export")
    assert r5.status_code == 200
    artifacts_dir = Path("data/assets/projects") / project_id / "artifacts"
    assert artifacts_dir.exists()
    files = [p.name for p in artifacts_dir.iterdir()]
    assert "notebook.ipynb" in files
    assert "report.txt" in files
