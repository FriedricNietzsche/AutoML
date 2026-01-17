"""
End-to-end demo flows for key scenarios.
- Cats vs Dogs (HF images)
- Insurance cost regression (HF tabular)
- Telco churn classification (HF tabular)
All tests use small samples to keep runtime low. Skips if OpenRouter key is missing or HF datasets unavailable.
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

# Keep runs fast by default
os.environ.setdefault("FAST_TEST", "1")
os.environ.setdefault("TRAIN_MAX_ROWS", "100")


@pytest.mark.skipif(not HAS_CREDS, reason="Requires OPENROUTER_API_KEY")
def test_cats_dogs_hf_image_flow():
    project_id = "demo-cats-dogs"
    project_dir = Path("data/assets/projects") / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir)

    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "Build a classifier for cats vs dogs"})
    assert r1.status_code == 200

    r2 = client.post(
        f"/api/projects/{project_id}/ingest/hf-images",
        json={
            "dataset": "Matias12f/cats_and_dogs",
            "split": "train",
            "max_images": 6,
            "image_field": "image",
            "label_field": "labels",
        },
    )
    if r2.status_code != 200:
        pytest.skip(f"HF cats/dogs dataset unavailable: {r2.text}")
    images_dir = project_dir / "images"
    assert images_dir.exists()
    assert any(images_dir.glob("*.png"))

    r3 = client.post(f"/api/projects/{project_id}/train/image", json={})
    assert r3.status_code == 200
    metrics = r3.json().get("metrics", {})
    assert "accuracy" in metrics

    # Report & export to ensure pipeline completes
    r4 = client.post(f"/api/projects/{project_id}/report", json={"metrics": metrics, "task_type": "classification", "target": "label"})
    assert r4.status_code == 200
    r5 = client.post(f"/api/projects/{project_id}/export")
    assert r5.status_code == 200


@pytest.mark.skipif(not HAS_CREDS, reason="Requires OPENROUTER_API_KEY")
def test_insurance_regression_flow():
    project_id = "demo-insurance"
    project_dir = Path("data/assets/projects") / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir)

    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "Predict insurance costs ethically"})
    assert r1.status_code == 200

    r2 = client.post(
        f"/api/projects/{project_id}/ingest/hf",
        json={"dataset": "hamidro/HealthyLife-Insurance-Charge-Prediction-v2", "split": "train", "max_rows": 200},
    )
    if r2.status_code != 200:
        pytest.skip(f"HF insurance dataset unavailable: {r2.text}")
    cols_raw = r2.json().get("columns", [])
    cols = [c.lower() for c in cols_raw]
    target = None
    for cand in ["charges", "expenses", "charge", "charges_usd"]:
        if cand in cols:
            target = cols_raw[cols.index(cand)]
            break
    if target is None:
        target = cols_raw[-1]

    r3 = client.post(
        f"/api/projects/{project_id}/train/tabular",
        json={"target": target, "task_type": "regression", "model_id": "linear"},
    )
    assert r3.status_code == 200
    metrics = r3.json().get("metrics", {})
    assert "rmse" in metrics

    r4 = client.post(
        f"/api/projects/{project_id}/report",
        json={"metrics": metrics, "task_type": "regression", "target": target, "source": {"hf_dataset": "insurance-cost-prediction"}},
    )
    assert r4.status_code == 200
    r5 = client.post(f"/api/projects/{project_id}/export")
    assert r5.status_code == 200


@pytest.mark.skipif(not HAS_CREDS, reason="Requires OPENROUTER_API_KEY")
def test_telco_churn_flow():
    project_id = "demo-telco"
    project_dir = Path("data/assets/projects") / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir)

    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "Analyze user activity logs to predict churn"})
    assert r1.status_code == 200

    r2 = client.post(
        f"/api/projects/{project_id}/ingest/hf",
        json={"dataset": "scikit-learn/churn-prediction", "split": "train", "max_rows": 200},
    )
    if r2.status_code != 200:
        pytest.skip(f"HF telco_churn dataset unavailable: {r2.text}")
    cols_raw = r2.json().get("columns", [])
    cols = [c.lower() for c in cols_raw]
    # pick a reasonable target (case sensitive if needed)
    target = None
    for cand in ["churn", "Churn"]:
        if cand in cols_raw:
            target = cand
            break
        if cand.lower() in cols:
            target = cols_raw[cols.index(cand.lower())]
            break
    if target is None:
        target = cols_raw[-1]

    r3 = client.post(
        f"/api/projects/{project_id}/train/tabular",
        json={"target": target, "task_type": "classification", "model_id": "logreg"},
    )
    assert r3.status_code == 200
    metrics = r3.json().get("metrics", {})
    assert "accuracy" in metrics

    r4 = client.post(
        f"/api/projects/{project_id}/report",
        json={"metrics": metrics, "task_type": "classification", "target": target, "source": {"hf_dataset": "d0rj/telco_churn"}},
    )
    assert r4.status_code == 200
    r5 = client.post(f"/api/projects/{project_id}/export")
    assert r5.status_code == 200
