"""
WebSocket connection manager (hub) for managing client connections per project.
"""
import asyncio
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime, timezone
import logging

from ..events.bus import event_bus
from ..events.schema import EventType, StageID, StageStatus
from ..orchestrator.conductor import conductor

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections grouped by project ID."""
    
    def __init__(self):
        # project_id -> set of WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        # Track unsubscribe functions for cleanup
        self._unsubscribers: Dict[WebSocket, callable] = {}
    
    async def connect(self, websocket: WebSocket, project_id: str) -> None:
        """
        Accept a new WebSocket connection and subscribe to project events.
        """
        await websocket.accept()
        # Small delay to ensure handshake is fully processed by browser before we flood it with data
        await asyncio.sleep(0.1)
        
        try:
            async with self._lock:
                if project_id not in self._connections:
                    self._connections[project_id] = set()
                self._connections[project_id].add(websocket)
            
            logger.info(f"WebSocket connected for project {project_id}. Total connections: {len(self._connections.get(project_id, set()))}")
            
            # Subscribe this connection to the event bus
            async def on_event(event: dict, _project_id: str = project_id):
                await self._send_to_socket(websocket, event, project_id=_project_id)
            
            unsubscribe = await event_bus.subscribe(project_id, on_event)
            self._unsubscribers[websocket] = unsubscribe
            
            # Send initial state
            await self._send_hello(websocket, project_id)
            await conductor.emit_current_status(project_id)
        except Exception as e:
            logger.error(f"Error during WebSocket initialization for {project_id}: {e}")
            raise

    
    async def disconnect(self, websocket: WebSocket, project_id: str) -> None:
        """
        Remove a WebSocket connection and unsubscribe from events.
        """
        async with self._lock:
            if project_id in self._connections:
                self._connections[project_id].discard(websocket)
                if not self._connections[project_id]:
                    del self._connections[project_id]
        
        # Unsubscribe from event bus
        if websocket in self._unsubscribers:
            unsubscribe = self._unsubscribers.pop(websocket)
            await unsubscribe()
        
        logger.info(f"WebSocket disconnected for project {project_id}")
    
    async def broadcast_to_project(self, project_id: str, message: dict) -> None:
        """
        Broadcast a message to all connections for a project.
        This goes through the event bus for consistency.
        """
        await event_bus.publish(project_id, message)
    
    async def _send_to_socket(self, websocket: WebSocket, message: dict, project_id: str | None = None) -> None:
        """Send a message to a specific WebSocket, handling errors."""
        try:
            await websocket.send_json(message)
        except WebSocketDisconnect:
            await self.disconnect(websocket, project_id or "unknown")
        except Exception as e:
            logger.error(f"Error sending to WebSocket: {e}")
    
    async def _send_hello(self, websocket: WebSocket, project_id: str) -> None:
        """Send a HELLO ping message on connection."""
        hello_event = event_bus.make_envelope(
            project_id=project_id,
            event_name=EventType.HELLO,
            payload={
                "message": "Connected to AutoML WebSocket",
                "project_id": project_id,
                "server_time": datetime.now(timezone.utc).isoformat()
            },
            stage_id=StageID.PARSE_INTENT,
            stage_status=StageStatus.PENDING,
            message_type="HELLO",
        )
        await self._send_to_socket(websocket, hello_event, project_id=project_id)
    
    def get_connection_count(self, project_id: str) -> int:
        """Get the number of active connections for a project."""
        return len(self._connections.get(project_id, set()))


# Global connection manager instance
manager = ConnectionManager()
