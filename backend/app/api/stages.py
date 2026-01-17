from fastapi import APIRouter, HTTPException

from ..events.schema import StageID, StageStatus
from ..orchestrator.conductor import conductor

router = APIRouter(prefix="/api/projects", tags=["stages"])


@router.get("/{project_id}/state")
async def get_project_state(project_id: str):
    """Return the current state snapshot for a project."""
    return await conductor.get_state(project_id)


@router.post("/{project_id}/confirm")
async def confirm_project_stage(project_id: str):
    """Advance to the next stage after user confirmation."""
    return await conductor.confirm(project_id)


@router.post("/{project_id}/stages/{stage_id}/status")
async def set_stage_status(project_id: str, stage_id: str, status: str, message: str | None = None):
    """Internal helper endpoint to force a stage status (useful for demos/tests)."""
    try:
        stage_enum = StageID(stage_id)
        status_enum = StageStatus(status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid stage id or status")
    return await conductor.transition_to(project_id, stage_enum, status=status_enum, message=message)
