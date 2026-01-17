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
    
    Connect to receive:
    - HELLO: Initial connection acknowledgment
    - STAGE_STATUS: Current stage and status updates
    - Various stage-specific events (PROMPT_PARSED, TRAIN_PROGRESS, etc.)
    
    Send messages to trigger actions (future implementation).
    """
    await manager.connect(websocket, project_id)
    
    try:
        while True:
            # Receive messages from client (for future chat/command handling)
            data = await websocket.receive_json()
            logger.info(f"Received from client {project_id}: {data}")
            
            # TODO: Handle incoming messages (chat, commands, etc.)
            # For now, just acknowledge receipt
            # This will be expanded in later tasks
            
    except WebSocketDisconnect:
        await manager.disconnect(websocket, project_id)
        logger.info(f"Client disconnected from project {project_id}")
    except Exception as e:
        logger.error(f"WebSocket error for project {project_id}: {e}")
        await manager.disconnect(websocket, project_id)
