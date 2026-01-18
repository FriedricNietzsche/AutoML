"""
Conductor - Manages the stage state machine for each project.
"""
import asyncio
from typing import Dict, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
import logging

from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus, stage_index

logger = logging.getLogger(__name__)

# Canonical stage order
STAGE_ORDER = [
    StageID.PARSE_INTENT,
    StageID.DATA_SOURCE,
    StageID.PROFILE_DATA,
    StageID.PREPROCESS,
    StageID.MODEL_SELECT,
    StageID.TRAIN,
    StageID.REVIEW_EDIT,
    StageID.EXPORT,
]


@dataclass
class StageState:
    """State of a single stage."""
    id: StageID
    status: StageStatus = StageStatus.PENDING
    message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    plan_pending: bool = False


@dataclass
class ProjectState:
    """Full state of a project's workflow."""
    project_id: str
    current_stage: StageID = StageID.PARSE_INTENT
    stages: Dict[StageID, StageState] = field(default_factory=dict)
    plan_pending: bool = False
    plan_approved: bool = False
    
    def __post_init__(self):
        # Initialize all stages if not provided
        if not self.stages:
            for stage_id in STAGE_ORDER:
                self.stages[stage_id] = StageState(id=stage_id)
            # Set first stage as in progress
            self.stages[StageID.PARSE_INTENT].status = StageStatus.PENDING


class Conductor:
    """
    Manages the workflow state machine for all projects.
    """
    
    def __init__(self):
        self._projects: Dict[str, ProjectState] = {}
        self._lock = asyncio.Lock()
    
    async def get_state(self, project_id: str) -> ProjectState:
        """Get or create project state."""
        async with self._lock:
            if project_id not in self._projects:
                self._projects[project_id] = ProjectState(project_id=project_id)
                logger.info(f"Initialized new project state for {project_id}")
            return self._projects[project_id]
    
    async def transition_to(
        self,
        project_id: str,
        stage_id: StageID,
        status: StageStatus = StageStatus.IN_PROGRESS,
        message: Optional[str] = None
    ) -> None:
        """
        Transition a project to a new stage/status.
        Emits STAGE_STATUS event.
        """
        state = await self.get_state(project_id)
        
        async with self._lock:
            stage_state = state.stages[stage_id]
            stage_state.status = status
            stage_state.message = message
            
            if status == StageStatus.IN_PROGRESS:
                stage_state.started_at = datetime.now(timezone.utc).isoformat()
                state.current_stage = stage_id
            elif status == StageStatus.COMPLETED:
                stage_state.completed_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Project {project_id}: Stage {stage_id.value} -> {status.value}")
        
        # Emit event
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload={
                "stage_id": stage_id.value,
                "status": status.value,
                "message": message or f"Stage {stage_id.value} is {status.value}",
            },
            stage_id=stage_id,
            stage_status=status,
        )
    
    async def waiting_for_confirmation(
        self,
        project_id: str,
        stage_id: StageID,
        summary: str,
        next_actions: list
    ) -> None:
        """
        Mark a stage as waiting for user confirmation.
        """
        state = await self.get_state(project_id)
        
        async with self._lock:
            state.stages[stage_id].status = StageStatus.WAITING_CONFIRMATION
            state.plan_pending = True
        
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.WAITING_CONFIRMATION,
            payload={
                "stage_id": stage_id.value,
                "summary": summary,
                "next_actions": next_actions,
            },
            stage_id=stage_id,
            stage_status=StageStatus.WAITING_CONFIRMATION,
        )
    
    async def confirm(self, project_id: str) -> StageID:
        """
        Confirm current stage and advance to next.
        Returns the new current stage.
        """
        state = await self.get_state(project_id)
        current_idx = STAGE_ORDER.index(state.current_stage)
        
        # Mark current as completed
        await self.transition_to(
            project_id,
            state.current_stage,
            StageStatus.COMPLETED,
            "Stage completed"
        )
        
        # Advance to next stage if not at end
        if current_idx < len(STAGE_ORDER) - 1:
            next_stage = STAGE_ORDER[current_idx + 1]
            await self.transition_to(
                project_id,
                next_stage,
                StageStatus.IN_PROGRESS,
                "Stage started"
            )
            
            async with self._lock:
                state.plan_pending = False
                state.plan_approved = True
            
            return next_stage
        
        return state.current_stage
    
    async def emit_current_status(self, project_id: str) -> None:
        """
        Emit the current stage status (used on WS connect).
        """
        state = await self.get_state(project_id)
        current = state.stages[state.current_stage]
        
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload={
                "stage_id": current.id.value,
                "status": current.status.value,
                "message": current.message or "Waiting for input",
            },
            stage_id=current.id,
            stage_status=current.status,
        )
    
    def get_state_snapshot(self, project_id: str) -> dict:
        """
        Get a JSON-serializable snapshot of project state.
        """
        if project_id not in self._projects:
            # Return default state
            return {
                "project_id": project_id,
                "current_stage": StageID.PARSE_INTENT.value,
                "stages": {
                    stage.value: {
                        "id": stage.value,
                        "index": stage_index(stage),
                        "status": StageStatus.PENDING.value,
                    }
                    for stage in STAGE_ORDER
                }
            }
        
        state = self._projects[project_id]
        return {
            "project_id": state.project_id,
            "current_stage": state.current_stage.value,
            "plan_pending": state.plan_pending,
            "stages": {
                stage_id.value: {
                    "id": stage_state.id.value,
                    "index": stage_index(stage_state.id),
                    "status": stage_state.status.value,
                    "message": stage_state.message,
                    "started_at": stage_state.started_at,
                    "completed_at": stage_state.completed_at,
                }
                for stage_id, stage_state in state.stages.items()
            }
        }


# Global conductor instance
conductor = Conductor()
