import asyncio
import os
import sys
import shutil
from pathlib import Path

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.api.assets import ASSET_ROOT  # noqa: E402
from app.ml.trainers.tabular_trainer import TabularTrainer, TrainConfig  # noqa: E402


def test_tabular_trainer_creates_metrics_and_confusion(tmp_path):
    project_id = "test_tabular_trainer"
    base = ASSET_ROOT / "projects" / project_id
    if base.exists():
        shutil.rmtree(base)

    df = pd.DataFrame(
        {
            "a": [1, 2, 1, 2, 3, 3],
            "b": ["x", "y", "x", "y", "x", "y"],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    trainer = TabularTrainer(
        TrainConfig(project_id=project_id, target="y", task_type="classification", steps=5, test_size=0.33)
    )

    result = asyncio.run(trainer.train(df))
    assert "metrics" in result
    assert "run_id" in result

    # Check artifact exists
    artifacts_dir = base / "artifacts"
    assert artifacts_dir.exists()
    confusion_files = list(artifacts_dir.glob("*confusion*.json"))
    assert confusion_files, "Confusion matrix asset not created"

    shutil.rmtree(base, ignore_errors=True)
