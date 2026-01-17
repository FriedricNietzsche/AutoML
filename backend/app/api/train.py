"""
Training endpoints for tabular and image pipelines.
Streams TRAIN_* events and transitions conductor stages.
"""
import asyncio
import os
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus
from app.ml.trainers.tabular_trainer import TabularTrainer, TrainConfig
from app.ml.trainers.image_trainer import ImageTrainer, ImageTrainConfig
from app.orchestrator.conductor import conductor
from app.api.assets import ASSET_ROOT

router = APIRouter(prefix="/api/projects", tags=["train"])
log = logging.getLogger(__name__)


def _project_dir(project_id: str) -> Path:
    path = ASSET_ROOT / "projects" / project_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Project data not found")
    return path


@router.post("/{project_id}/train/tabular")
async def train_tabular(
    project_id: str,
    target: str = Body(..., embed=True),
    task_type: str = Body("classification", embed=True),
    model_id: str = Body("auto", embed=True),
):
    project_dir = _project_dir(project_id)
    csv_files = list(project_dir.glob("*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail="No CSV found for project")
    df = pd.read_csv(csv_files[0])
    # Guardrail: cap rows to keep training fast in demos/e2e (especially wide/large datasets like MNIST CSV).
    max_rows = int(os.getenv("TRAIN_MAX_ROWS", "50"))
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    log.info("Training tabular model: project=%s rows=%s cols=%s task=%s model=%s", project_id, len(df), len(df.columns), task_type, model_id)

    trainer = TabularTrainer(TrainConfig(project_id=project_id, target=target, task_type=task_type, model_id=model_id))
    await conductor.transition_to(project_id, StageID.TRAIN, StageStatus.IN_PROGRESS, "Training started")
    result = await trainer.train(df)
    await conductor.transition_to(project_id, StageID.TRAIN, StageStatus.COMPLETED, "Training complete")
    await conductor.transition_to(project_id, StageID.REVIEW_EDIT, StageStatus.IN_PROGRESS, "Review artifacts")

    return result


@router.post("/{project_id}/train/image")
async def train_image(project_id: str, data_subdir: Optional[str] = Body(None, embed=True)):
    project_dir = _project_dir(project_id)
    data_dir = project_dir / (data_subdir or "images")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Image data directory not found")

    trainer = ImageTrainer(ImageTrainConfig(project_id=project_id, data_dir=data_dir))
    await conductor.transition_to(project_id, StageID.TRAIN, StageStatus.IN_PROGRESS, "Training started")
    result = await trainer.train()
    await conductor.transition_to(project_id, StageID.TRAIN, StageStatus.COMPLETED, "Training complete")
    await conductor.transition_to(project_id, StageID.REVIEW_EDIT, StageStatus.IN_PROGRESS, "Review artifacts")

    return result
