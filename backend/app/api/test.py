"""
Test endpoints for verifying WebSocket and event bus functionality.
"""
from fastapi import APIRouter, HTTPException

from ..events.bus import event_bus
from ..events.schema import EventType, StageID, StageStatus
from ..ws.hub import manager

router = APIRouter(prefix="/api/test", tags=["test"])


@router.post("/emit/{project_id}")
async def emit_test_event(project_id: str, message: str = "Test event"):
    """
    Emit a test LOG_LINE event to all WebSocket clients for a project.
    Useful for verifying the event bus and WebSocket setup.
    """
    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.LOG_LINE,
        payload={
            "run_id": "test",
            "level": "info",
            "text": message,
            "source": "test_endpoint",
        },
        stage_id=StageID.PARSE_INTENT,
        stage_status=StageStatus.IN_PROGRESS,
    )
    
    return {
        "status": "emitted",
        "project_id": project_id,
        "connections": manager.get_connection_count(project_id),
        "subscribers": event_bus.get_subscriber_count(project_id)
    }


@router.post("/stage-update/{project_id}")
async def emit_stage_update(project_id: str, stage_id: str, status: str):
    """
    Emit a stage status update event.
    """
    try:
        stage_enum = StageID(stage_id)
        status_enum = StageStatus(status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid stage_id or status")

    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.STAGE_STATUS,
        payload={
            "stage_id": stage_enum.value,
            "status": status_enum.value,
            "message": f"Stage {stage_enum.value} is now {status_enum.value}",
        },
        stage_id=stage_enum,
        stage_status=status_enum,
    )
    
    return {
        "status": "emitted",
        "stage_id": stage_enum.value,
        "stage_status": status_enum.value
    }


@router.get("/connections/{project_id}")
async def get_connections(project_id: str):
    """Get the number of active WebSocket connections for a project."""
    return {
        "project_id": project_id,
        "connections": manager.get_connection_count(project_id),
        "subscribers": event_bus.get_subscriber_count(project_id)
    }
