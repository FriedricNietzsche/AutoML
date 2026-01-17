"""
In-memory event bus for pub/sub within the backend.
Allows agents/services to publish events that get broadcast to WebSocket clients.
"""
import asyncio
import time
from typing import Callable, Dict, List, Any, Optional
from collections import defaultdict
import logging

from .schema import (
    EventPayload,
    EventType,
    Stage,
    StageID,
    StageStatus,
    WSEnvelope,
    stage_index,
)

logger = logging.getLogger(__name__)


class EventBus:
    """Simple in-memory pub/sub event bus."""
    
    def __init__(self):
        # project_id -> list of async callback functions
        self._subscribers: Dict[str, List[Callable[[dict], Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._seq_counters: Dict[str, int] = defaultdict(int)

    def next_seq(self, project_id: str) -> int:
        """Reserve the next sequence number for a project."""
        self._seq_counters[project_id] += 1
        return self._seq_counters[project_id]

    def make_envelope(
        self,
        project_id: str,
        event_name: EventType,
        payload: Dict[str, Any],
        *,
        stage_id: StageID = StageID.PARSE_INTENT,
        stage_status: StageStatus = StageStatus.PENDING,
        stage_index_override: Optional[int] = None,
        message_type: str = "EVENT",
        version: int = 1,
        seq: Optional[int] = None,
        ts: Optional[int] = None,
    ) -> dict:
        """
        Build a WebSocket envelope with defaults aligned to the contract.
        """
        envelope = WSEnvelope(
            v=version,
            type=message_type,
            project_id=project_id,
            seq=seq if seq is not None else self.next_seq(project_id),
            ts=ts if ts is not None else int(time.time() * 1000),
            stage=Stage(
                id=stage_id,
                index=stage_index_override if stage_index_override is not None else stage_index(stage_id),
                status=stage_status,
            ),
            event=EventPayload(name=event_name, payload=payload),
        )
        return envelope.model_dump()
    
    async def subscribe(self, project_id: str, callback: Callable[[dict], Any]) -> Callable:
        """
        Subscribe to events for a specific project.
        Returns an unsubscribe function.
        """
        async with self._lock:
            self._subscribers[project_id].append(callback)
            logger.info(f"Subscriber added for project {project_id}. Total: {len(self._subscribers[project_id])}")
        
        async def unsubscribe():
            async with self._lock:
                if callback in self._subscribers[project_id]:
                    self._subscribers[project_id].remove(callback)
                    logger.info(f"Subscriber removed for project {project_id}. Total: {len(self._subscribers[project_id])}")
                # Clean up empty lists
                if not self._subscribers[project_id]:
                    del self._subscribers[project_id]
        
        return unsubscribe
    
    async def publish(self, project_id: str, event: dict) -> None:
        """
        Publish an event to all subscribers of a project.
        Event should follow the WSEnvelope schema.
        """
        if isinstance(event, WSEnvelope):
            envelope = event.model_dump()
        else:
            envelope = dict(event)

        # Fill envelope defaults if missing (helps when callers hand us raw payloads)
        envelope.setdefault("v", 1)
        envelope.setdefault("type", "EVENT")
        envelope.setdefault("seq", self.next_seq(project_id))
        envelope.setdefault("ts", int(time.time() * 1000))
        envelope.setdefault(
            "stage",
            Stage(
                id=StageID.PARSE_INTENT,
                index=stage_index(StageID.PARSE_INTENT),
                status=StageStatus.PENDING,
            ).model_dump(),
        )
        envelope.setdefault("event", {})

        async with self._lock:
            subscribers = list(self._subscribers.get(project_id, []))
        
        if not subscribers:
            logger.debug(f"No subscribers for project {project_id}, event dropped: {envelope.get('event')}")
            return
        
        logger.info(
            f"Publishing event {envelope.get('event', {}).get('name')} "
            f"to {len(subscribers)} subscribers for project {project_id}"
        )
        
        # Broadcast to all subscribers concurrently
        await asyncio.gather(
            *[self._safe_callback(callback, envelope) for callback in subscribers],
            return_exceptions=True
        )

    async def publish_event(
        self,
        project_id: str,
        event_name: EventType,
        payload: Dict[str, Any],
        *,
        stage_id: StageID = StageID.PARSE_INTENT,
        stage_status: StageStatus = StageStatus.PENDING,
        stage_index_override: Optional[int] = None,
        message_type: str = "EVENT",
        version: int = 1,
    ) -> None:
        """Helper to build + publish a typed envelope in one call."""
        envelope = self.make_envelope(
            project_id=project_id,
            event_name=event_name,
            payload=payload,
            stage_id=stage_id,
            stage_status=stage_status,
            stage_index_override=stage_index_override,
            message_type=message_type,
            version=version,
        )
        await self.publish(project_id, envelope)
    
    async def _safe_callback(self, callback: Callable[[dict], Any], event: dict) -> None:
        """Safely execute a callback, catching any exceptions."""
        try:
            result = callback(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in event callback: {e}")
    
    def get_subscriber_count(self, project_id: str) -> int:
        """Get the number of subscribers for a project."""
        return len(self._subscribers.get(project_id, []))


# Global event bus instance
event_bus = EventBus()
