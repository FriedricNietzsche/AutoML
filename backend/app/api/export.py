"""
Reporter and export endpoints.
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Any
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import APIRouter, Body, HTTPException

from app.agents.reporter import ReporterAgent
from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus
from app.ml.artifacts import save_text_asset, save_notebook_asset, asset_url, project_dir
from app.api.assets import ASSET_ROOT
from app.orchestrator.conductor import conductor

router = APIRouter(prefix="/api/projects", tags=["report-export"])


@router.post("/{project_id}/report")
async def generate_report(project_id: str, context: Dict[str, Any] = Body(default_factory=dict)):
    agent = ReporterAgent()
    outputs = agent.generate(context)

    notebook_path = save_notebook_asset(project_id, "artifacts/notebook.ipynb", outputs["notebook_nb"])
    report_path = save_text_asset(project_id, "artifacts/report.txt", outputs["report"])

    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.NOTEBOOK_READY,
        payload={"asset_url": asset_url(notebook_path)},
        stage_id=StageID.REVIEW_EDIT,
        stage_status=StageStatus.IN_PROGRESS,
    )
    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.REPORT_READY,
        payload={"asset_url": asset_url(report_path)},
        stage_id=StageID.REVIEW_EDIT,
        stage_status=StageStatus.IN_PROGRESS,
    )

    await conductor.transition_to(project_id, StageID.REVIEW_EDIT, StageStatus.COMPLETED, "Report ready")
    await conductor.transition_to(project_id, StageID.EXPORT, StageStatus.IN_PROGRESS, "Prepare export")

    return {"notebook": asset_url(notebook_path), "report": asset_url(report_path)}


@router.post("/{project_id}/export")
async def export_bundle(project_id: str):
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="Project data not found")
    zip_path = base / "export.zip"
    contents = []
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(base):
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(base)
                zf.write(full, rel)
                contents.append(str(rel))

    # Compute checksum
    import hashlib

    sha = hashlib.sha256()
    with zip_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    checksum = sha.hexdigest()

    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.EXPORT_READY,
        payload={"asset_url": asset_url(zip_path), "contents": contents, "checksum": checksum},
        stage_id=StageID.EXPORT,
        stage_status=StageStatus.COMPLETED,
    )
    await conductor.transition_to(project_id, StageID.EXPORT, StageStatus.COMPLETED, "Export ready")
    return {"export": asset_url(zip_path)}
