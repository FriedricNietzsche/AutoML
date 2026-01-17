import asyncio
from typing import Any, Dict, List, Optional, Tuple

from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus, STAGE_SEQUENCE, stage_index

StageState = Dict[str, Any]
ProjectState = Dict[str, Any]


class Conductor:
    """
    Lightweight in-memory conductor that tracks per-project stage state.
    Emits STAGE_STATUS and WAITING_CONFIRMATION events via the shared event bus.
    """

    def __init__(self) -> None:
        self._projects: Dict[str, ProjectState] = {}
        self._lock = asyncio.Lock()

    def _default_project(self, project_id: str) -> ProjectState:
        stages: Dict[StageID, StageState] = {
            stage_id: {"id": stage_id, "status": StageStatus.PENDING, "message": None}
            for stage_id in STAGE_SEQUENCE
        }
        stages[StageID.PARSE_INTENT]["status"] = StageStatus.IN_PROGRESS
        return {
            "project_id": project_id,
            "current_stage": StageID.PARSE_INTENT,
            "stages": stages,
            "waiting_confirmation": None,
        }

    def _serialize_state(self, state: ProjectState) -> Dict[str, Any]:
        current_stage_id: StageID = state["current_stage"]
        stages: Dict[StageID, StageState] = state["stages"]
        return {
            "project_id": state["project_id"],
            "current_stage": {
                "id": current_stage_id.value,
                "index": stage_index(current_stage_id),
                "status": stages[current_stage_id]["status"].value,
                "message": stages[current_stage_id].get("message"),
            },
            "stages": [
                {
                    "id": stage_id.value,
                    "index": stage_index(stage_id),
                    "status": stage_state["status"].value,
                    "message": stage_state.get("message"),
                }
                for stage_id in STAGE_SEQUENCE
                for stage_state in [stages[stage_id]]
            ],
            "waiting_confirmation": state.get("waiting_confirmation"),
        }

    async def get_state(self, project_id: str) -> Dict[str, Any]:
        async with self._lock:
            state = self._projects.get(project_id) or self._default_project(project_id)
            self._projects[project_id] = state
            return self._serialize_state(state)

    async def transition_to(
        self,
        project_id: str,
        stage_id: StageID,
        status: StageStatus = StageStatus.IN_PROGRESS,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            state = self._projects.get(project_id) or self._default_project(project_id)
            self._projects[project_id] = state

            stage_state = state["stages"][stage_id]
            stage_state["status"] = status
            if message:
                stage_state["message"] = message
            state["current_stage"] = stage_id
            state["waiting_confirmation"] = None
            payload = {
                "stage_id": stage_id.value,
                "status": status.value,
                "message": message or stage_state.get("message") or "",
            }

        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload=payload,
            stage_id=stage_id,
            stage_status=status,
        )
        return self._serialize_state(state)

    async def waiting_for_confirmation(
        self,
        project_id: str,
        stage_id: StageID,
        summary: str,
        next_actions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        actions = next_actions or []
        async with self._lock:
            state = self._projects.get(project_id) or self._default_project(project_id)
            self._projects[project_id] = state
            state["waiting_confirmation"] = {
                "stage_id": stage_id.value,
                "summary": summary,
                "next_actions": actions,
            }
            stage_state = state["stages"][stage_id]
            stage_state["status"] = StageStatus.WAITING_CONFIRMATION
            stage_state["message"] = summary
            payload = {
                "stage_id": stage_id.value,
                "summary": summary,
                "next_actions": actions,
            }

        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.WAITING_CONFIRMATION,
            payload=payload,
            stage_id=stage_id,
            stage_status=StageStatus.WAITING_CONFIRMATION,
        )
        return self._serialize_state(state)

    async def confirm(self, project_id: str) -> Dict[str, Any]:
        async with self._lock:
            state = self._projects.get(project_id) or self._default_project(project_id)
            self._projects[project_id] = state
            current_stage: StageID = state["current_stage"]
            stages: Dict[StageID, StageState] = state["stages"]

            updates: List[Tuple[StageID, StageStatus, str]] = []
            if stages[current_stage]["status"] != StageStatus.COMPLETED:
                stages[current_stage]["status"] = StageStatus.COMPLETED
                updates.append((current_stage, StageStatus.COMPLETED, "Stage confirmed"))

            current_idx = stage_index(current_stage)
            next_stage: Optional[StageID] = None
            if current_idx + 1 < len(STAGE_SEQUENCE):
                next_stage = STAGE_SEQUENCE[current_idx + 1]
                stages[next_stage]["status"] = StageStatus.IN_PROGRESS
                state["current_stage"] = next_stage
                updates.append((next_stage, StageStatus.IN_PROGRESS, "Advancing to next stage"))
            state["waiting_confirmation"] = None
            snapshot = self._serialize_state(state)

        for stage_id, status, message in updates:
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.STAGE_STATUS,
                payload={
                    "stage_id": stage_id.value,
                    "status": status.value,
                    "message": message,
                },
                stage_id=stage_id,
                stage_status=status,
            )
        return snapshot

    async def emit_current_status(self, project_id: str) -> None:
        async with self._lock:
            state = self._projects.get(project_id) or self._default_project(project_id)
            self._projects[project_id] = state
            current_stage: StageID = state["current_stage"]
            stage_state = state["stages"][current_stage]
            payload = {
                "stage_id": current_stage.value,
                "status": stage_state["status"].value,
                "message": stage_state.get("message") or "",
            }

        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload=payload,
            stage_id=current_stage,
            stage_status=stage_state["status"],
        )


# Global conductor instance
conductor = Conductor()
