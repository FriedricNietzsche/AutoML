"""
Stages API - Project state and stage management.
"""
from fastapi import APIRouter, HTTPException

from ..orchestrator.conductor import conductor
from ..events.schema import StageID, StageStatus

router = APIRouter(prefix="/api/projects", tags=["stages"])


@router.get("/{project_id}/state")
async def get_project_state(project_id: str):
    """
    Get the current state snapshot of a project.
    Used by frontend to hydrate on load.
    """
    return conductor.get_state_snapshot(project_id)


@router.post("/{project_id}/confirm")
async def confirm_stage(project_id: str):
    """
    Confirm the current stage and advance to the next.
    """
    try:
        new_stage = await conductor.confirm(project_id)
        return {
            "status": "confirmed",
            "new_stage": new_stage.value,
            "project_id": project_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_id}/transition")
async def transition_stage(
    project_id: str,
    stage_id: str,
    status: str = "IN_PROGRESS",
    message: str = None
):
    """
    Manually transition a project to a specific stage/status.
    """
    try:
        stage_enum = StageID(stage_id)
        status_enum = StageStatus(status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid stage_id or status")
    
    await conductor.transition_to(project_id, stage_enum, status_enum, message)
    return {
        "status": "transitioned",
        "stage_id": stage_enum.value,
        "stage_status": status_enum.value,
    }
