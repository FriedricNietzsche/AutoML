from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any
from pathlib import Path

from ..events.schema import StageID, StageStatus
from ..orchestrator.conductor import conductor
from ..orchestrator.pipeline import orchestrator

router = APIRouter(prefix="/api/projects", tags=["stages"])


@router.get("/{project_id}/state")
async def get_project_state(project_id: str):
    """Return the current state snapshot for a project."""
    return await conductor.get_state(project_id)


@router.post("/{project_id}/confirm")
async def confirm_project_stage(
    project_id: str, 
    selection: Optional[Dict[str, Any]] = Body(default=None)
):
    """
    Advance to the next stage after user confirmation.
    Automatically triggers the orchestrator to execute the next stage.
    
    Args:
        project_id: The project ID
        selection: Optional user selection (e.g., {"dataset_id": "123", "model_id": "rf"})
    """
    print(f"\n[API] Confirm stage for project: {project_id}")
    if selection:
        print(f"[API] User selection: {selection}")
    
    # Handle confirmation and auto-execute next stage
    await orchestrator.handle_confirmation(project_id, selection)
    
    # Return updated state
    return await conductor.get_state(project_id)


@router.post("/{project_id}/stages/{stage_id}/status")
async def set_stage_status(project_id: str, stage_id: str, status: str, message: str | None = None):
    """Internal helper endpoint to force a stage status (useful for demos/tests)."""
    try:
        stage_enum = StageID(stage_id)
        status_enum = StageStatus(status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid stage id or status")
    return await conductor.transition_to(project_id, stage_enum, status=status_enum, message=message)


@router.get("/{project_id}/export/download")
async def download_export(project_id: str, timestamp: str | None = None):
    """Download the exported model package (ZIP file).
    
    Args:
        project_id: The project ID
        timestamp: Optional timestamp (YYYYMMDD_HHMMSS). If not provided, downloads the latest export.
    """
    project_dir = Path(f"data/assets/projects/{project_id}")
    
    if timestamp:
        # Download specific export
        export_path = project_dir / f"{project_id}_export_{timestamp}.zip"
    else:
        # Find the latest export
        export_files = list(project_dir.glob(f"{project_id}_export_*.zip"))
        if not export_files:
            raise HTTPException(status_code=404, detail="No export packages found. Run export stage first.")
        # Sort by filename (timestamp is in the filename, so alphabetical = chronological)
        export_path = sorted(export_files)[-1]
    
    if not export_path.exists():
        raise HTTPException(status_code=404, detail=f"Export package not found: {export_path.name}")
    
    return FileResponse(
        path=export_path,
        filename=export_path.name,
        media_type="application/zip"
    )

