import asyncio
import os
import sys
import shutil
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.api.assets import ASSET_ROOT  # noqa: E402
from app.ml.trainers.tabular_trainer import TabularTrainer, TrainConfig  # noqa: E402


def reset_project(project_id: str):
    base = ASSET_ROOT / "projects" / project_id
    if base.exists():
        shutil.rmtree(base)


def test_tabular_classification_random_forest():
    project_id = "test_tabular_classification_rf"
    reset_project(project_id)
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "cat": ["a", "b", "a", "b", "a", "b"],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )
    trainer = TabularTrainer(TrainConfig(project_id=project_id, target="label", task_type="classification", steps=5, test_size=0.33))
    result = asyncio.run(trainer.train(df))
    assert result["metrics"]["accuracy"] >= 0.5
    assert "confusion" in result["artifacts"]


def test_tabular_regression_residuals():
    project_id = "test_tabular_regression"
    reset_project(project_id)
    df = pd.DataFrame(
        {
            "feature": [0, 1, 2, 3, 4, 5],
            "target": [0.0, 1.1, 1.9, 3.2, 3.8, 5.1],
        }
    )
    trainer = TabularTrainer(TrainConfig(project_id=project_id, target="target", task_type="regression", steps=5, test_size=0.33))
    result = asyncio.run(trainer.train(df))
    assert "rmse" in result["metrics"]
    assert "residuals" in result["artifacts"]

