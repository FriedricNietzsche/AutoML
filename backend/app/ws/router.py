"""
WebSocket router for real-time project updates.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from .hub import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/projects/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """
    WebSocket endpoint for real-time project updates.
    """
    try:
        # Step 1: Initialize connection and subscribe to events
        await manager.connect(websocket, project_id)
        
        # Step 2: Keep alive / Receive loop
        try:
            while True:
                # We use receive_text() to wait for messages or keep connection open.
                # Most clients will only listen, but this keeps the task from completing.
                data = await websocket.receive_text()
                logger.info(f"Received from client {project_id}: {data}")
        except WebSocketDisconnect:
            logger.info(f"Client disconnected gracefully from {project_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket loop for {project_id}: {e}")
    except Exception as e:
        logger.error(f"Failed to establish WebSocket for {project_id}: {e}")
    finally:
        # Ensure cleanup on any exit
        try:
            await manager.disconnect(websocket, project_id)
        except Exception as e:
            logger.debug(f"Cleanup error for {project_id}: {e}")

