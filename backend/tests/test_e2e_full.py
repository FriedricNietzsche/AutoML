"""
Full end-to-end smoke test (HTTP only, not WS assertions).
Flow: parse -> ingest -> train -> report -> export.
Requires env: OPENROUTER_API_KEY (optional), KAGGLE_USERNAME/KAGGLE_KEY for Kaggle ingest.
Note: image/time-series/text flows are not included here; this covers tabular for determinism.
"""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append("backend")
from app.main import app  # noqa: E402

client = TestClient(app)


HAS_KAGGLE = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))


@pytest.mark.skipif(not HAS_KAGGLE, reason="Requires Kaggle creds")
def test_full_tabular_pipeline():
    project_id = "e2e-full-demo"

    # 1) Parse intent
    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "predict house prices from features"})
    assert r1.status_code == 200
    state = client.get(f"/api/projects/{project_id}/state").json()
    assert state["stages"][0]["status"] in {"COMPLETED", "IN_PROGRESS"}

    # 2) Ingest dataset (using iris for a small deterministic tabular example)
    r2 = client.post(f"/api/projects/{project_id}/ingest/kaggle", json={"dataset": "uciml/iris"})
    assert r2.status_code == 200
    assert "Species" in r2.json().get("columns", [])
    state = client.get(f"/api/projects/{project_id}/state").json()
    # DATA_SOURCE should be completed, PROFILE_DATA in progress
    stage_map = {s["id"]: s for s in state["stages"]}
    assert stage_map["DATA_SOURCE"]["status"] == "COMPLETED"
    assert stage_map["PROFILE_DATA"]["status"] in {"IN_PROGRESS", "COMPLETED"}

    # 3) Train tabular
    r3 = client.post(f"/api/projects/{project_id}/train/tabular", json={"target": "Species", "task_type": "classification"})
    assert r3.status_code == 200
    metrics = r3.json().get("metrics", {})
    artifacts = r3.json().get("artifacts", {})
    assert "accuracy" in metrics
    assert "confusion" in artifacts
    state = client.get(f"/api/projects/{project_id}/state").json()
    stage_map = {s["id"]: s for s in state["stages"]}
    assert stage_map["TRAIN"]["status"] == "COMPLETED"
    assert stage_map["REVIEW_EDIT"]["status"] in {"IN_PROGRESS", "COMPLETED"}

    # 4) Report
    context = {"profile": r2.json(), "metrics": metrics}
    r4 = client.post(f"/api/projects/{project_id}/report", json=context)
    assert r4.status_code == 200
    state = client.get(f"/api/projects/{project_id}/state").json()
    stage_map = {s["id"]: s for s in state["stages"]}
    assert stage_map["REVIEW_EDIT"]["status"] == "COMPLETED"
    assert stage_map["EXPORT"]["status"] in {"IN_PROGRESS", "COMPLETED"}

    # 5) Export
    r5 = client.post(f"/api/projects/{project_id}/export")
    assert r5.status_code == 200
    export_url = r5.json().get("export")
    assert export_url
    state = client.get(f"/api/projects/{project_id}/state").json()
    stage_map = {s["id"]: s for s in state["stages"]}
    assert stage_map["EXPORT"]["status"] == "COMPLETED"

    artifacts_dir = Path("data/assets/projects") / project_id / "artifacts"
    assert artifacts_dir.exists()
    files = [p.name for p in artifacts_dir.iterdir()]
    assert any("confusion" in f for f in files)
    assert "notebook.ipynb" in files
    assert "report.txt" in files


@pytest.mark.skipif(not HAS_KAGGLE, reason="Requires Kaggle creds")
def test_full_house_prices_pipeline():
    project_id = "e2e-house-demo"
    project_dir = Path("data/assets/projects") / project_id
    if project_dir.exists():
        import shutil
        shutil.rmtree(project_dir)

    # 1) Parse intent
    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "Build me a model to predict house prices"})
    assert r1.status_code == 200

    # 2) Ingest a small public housing dataset (public and ungated)
    slug = "dansbecker/melbourne-housing-snapshot"
    file_name = "melb_data.csv"
    r2 = client.post(
        f"/api/projects/{project_id}/ingest/kaggle",
        json={"dataset": slug, "file_name": file_name},
    )
    assert r2.status_code == 200
    assert "price" in [c.lower() for c in r2.json().get("columns", [])]

    # 3) Train regression
    r3 = client.post(
        f"/api/projects/{project_id}/train/tabular",
        json={"target": "Price", "task_type": "regression"},
    )
    assert r3.status_code == 200
    metrics = r3.json().get("metrics", {})
    artifacts = r3.json().get("artifacts", {})
    assert "rmse" in metrics
    assert "residuals" in artifacts

    # 4) Report with source metadata for notebook
    context = {
        "profile": r2.json(),
        "metrics": metrics,
        "task_type": "regression",
        "target": "Price",
        "source": {"kaggle_slug": slug, "kaggle_file": file_name},
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


@pytest.mark.skipif(not HAS_KAGGLE, reason="Requires Kaggle creds")
def test_auto_dataset_flow():
    """
    End-to-end flow using auto dataset discovery (slug hint) to validate full pipeline.
    """
    project_id = "e2e-auto-demo"
    project_dir = Path("data/assets/projects") / project_id
    if project_dir.exists():
        import shutil
        shutil.rmtree(project_dir)

    # 1) Parse intent
    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "Build a classifier for iris flowers"})
    assert r1.status_code == 200

    # 2) Auto-ingest using explicit slug (public, ungated)
    slug = "uciml/iris"
    r2 = client.post(
        f"/api/projects/{project_id}/ingest/auto",
        json={"dataset_hint": slug, "file_name": "Iris.csv"},
    )
    assert r2.status_code == 200
    assert "species" in [c.lower() for c in r2.json().get("columns", [])]

    # 3) Train classification
    r3 = client.post(
        f"/api/projects/{project_id}/train/tabular",
        json={"target": "Species", "task_type": "classification"},
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
        "target": "Species",
        "source": {"kaggle_slug": slug, "kaggle_file": "Iris.csv"},
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


@pytest.mark.skipif(not HAS_KAGGLE, reason="Requires Kaggle creds")
def test_full_digits_classification():
    """
    End-to-end flow for handwritten digit classification using public Kaggle CSV (MNIST in CSV).
    """
    project_id = "e2e-full-digits"
    project_dir = Path("data/assets/projects") / project_id
    if project_dir.exists():
        import shutil
        shutil.rmtree(project_dir)

    # 1) Parse intent
    r1 = client.post(f"/api/projects/{project_id}/parse", json={"prompt": "Build a classifier for handwritten digits"})
    assert r1.status_code == 200

    # 2) Auto-ingest using a public MNIST CSV dataset
    slug = "oddrationale/mnist-in-csv"
    file_name = "mnist_train.csv"
    r2 = client.post(
        f"/api/projects/{project_id}/ingest/auto",
        json={"dataset_hint": slug, "file_name": file_name},
    )
    assert r2.status_code == 200
    assert "label" in [c.lower() for c in r2.json().get("columns", [])]

    # 3) Train classification (target column 'label')
    r3 = client.post(
        f"/api/projects/{project_id}/train/tabular",
        json={"target": "label", "task_type": "classification"},
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
        "source": {"kaggle_slug": slug, "kaggle_file": file_name},
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
