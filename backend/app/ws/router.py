"""
WebSocket router for real-time project updates.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
from typing import Dict, Any

from .hub import manager
from ..events.bus import event_bus
from ..events.schema import EventType, StageID, StageStatus
from ..orchestrator.conductor import conductor
from ..agents.chat_agent import get_chat_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


async def handle_client_message(project_id: str, data: Dict[str, Any]) -> None:
    """
    Handle incoming messages from the WebSocket client.
    
    Supported message types:
    - chat: User chat messages/change requests
    - confirm: Confirm current stage and advance
    - command: Execute specific commands
    - ping: Keepalive ping
    """
    msg_type = data.get("type", "unknown")
    
    if msg_type == "chat":
        # Handle chat/change request messages
        text = data.get("text", "").strip()
        if text:
            logger.info(f"[{project_id}] User chat: {text}")
            
            # Get project context for the chat agent
            state = await conductor.get_state(project_id)
            context = {
                "current_stage": state.get("current_stage", {}).get("id", "PARSE_INTENT"),
                "stage_status": state.get("current_stage", {}).get("status", "IN_PROGRESS"),
                "task_type": "classification",  # TODO: Get from project state
                "target_column": "unknown",      # TODO: Get from project state
                "current_model": "not selected", # TODO: Get from project state
            }
            
            # Process message through LangChain chat agent
            try:
                chat_agent = get_chat_agent()
                response = await chat_agent.process_message(project_id, text, context)
                
                logger.info(f"[{project_id}] Chat agent response: {response.message} (intent={response.intent})")
                
                # Send assistant response as LOG_LINE event
                await event_bus.publish_event(
                    project_id=project_id,
                    event_name=EventType.LOG_LINE,
                    payload={
                        "run_id": "chat",
                        "level": "info",
                        "text": f"🤖 Assistant: {response.message}",
                        "source": "chat_agent",
                    },
                    stage_id=conductor._get_current_stage_id(project_id),
                    stage_status=StageStatus.IN_PROGRESS,
                )
                
                # If agent suggests an action, emit it as well
                if response.suggested_action and response.suggested_action != "none":
                    await event_bus.publish_event(
                        project_id=project_id,
                        event_name=EventType.LOG_LINE,
                        payload={
                            "run_id": "chat",
                            "level": "warn",
                            "text": f"💡 Suggested action: {response.suggested_action}",
                            "source": "chat_agent",
                        },
                        stage_id=conductor._get_current_stage_id(project_id),
                        stage_status=StageStatus.IN_PROGRESS,
                    )
                    
                    # TODO: Route suggested actions to appropriate handlers
                    # For now, just log the suggestion
                    if response.suggested_action == "retrain":
                        logger.info(f"[{project_id}] User requested retraining with params: {response.parameters}")
                    elif response.suggested_action == "export":
                        logger.info(f"[{project_id}] User requested export")
                    elif response.suggested_action == "adjust_params":
                        logger.info(f"[{project_id}] User requested parameter adjustment: {response.parameters}")
                
            except Exception as e:
                logger.error(f"[{project_id}] Chat agent failed: {e}", exc_info=True)
                # Fallback: echo user message
                await event_bus.publish_event(
                    project_id=project_id,
                    event_name=EventType.LOG_LINE,
                    payload={
                        "run_id": "chat",
                        "level": "info",
                        "text": f"User: {text}",
                        "source": "chat",
                    },
                    stage_id=conductor._get_current_stage_id(project_id),
                    stage_status=StageStatus.IN_PROGRESS,
                )
    
    elif msg_type == "confirm":
        # User confirms current stage - advance pipeline
        logger.info(f"[{project_id}] User confirmed stage via WebSocket")
        try:
            await conductor.confirm(project_id)
        except Exception as e:
            logger.error(f"[{project_id}] Confirm failed: {e}")
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.LOG_LINE,
                payload={
                    "run_id": "system",
                    "level": "error",
                    "text": f"Confirmation failed: {str(e)}",
                    "source": "websocket",
                },
                stage_id=conductor._get_current_stage_id(project_id),
                stage_status=StageStatus.FAILED,
            )
    
    elif msg_type == "command":
        # Execute specific command
        command = data.get("command", "")
        args = data.get("args", {})
        logger.info(f"[{project_id}] Command received: {command} with args: {args}")
        
        # Handle specific commands
        if command == "restart_stage":
            stage_id = args.get("stage_id")
            if stage_id:
                try:
                    stage_enum = StageID(stage_id)
                    await conductor.transition_to(
                        project_id, 
                        stage_enum, 
                        status=StageStatus.IN_PROGRESS,
                        message="Stage restarted by user"
                    )
                except ValueError:
                    logger.error(f"Invalid stage_id: {stage_id}")
        
        elif command == "cancel_run":
            # Cancel current training run
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.LOG_LINE,
                payload={
                    "run_id": "system",
                    "level": "warn",
                    "text": "Run cancellation requested",
                    "source": "websocket",
                },
                stage_id=StageID.TRAIN,
                stage_status=StageStatus.IN_PROGRESS,
            )
            # TODO: Implement actual cancellation logic
        
        else:
            logger.warning(f"[{project_id}] Unknown command: {command}")
    
    elif msg_type == "ping":
        # Keepalive ping - respond with pong
        await manager.broadcast_to_project(project_id, {
            "type": "pong",
            "ts": data.get("ts"),
            "server_ts": event_bus.next_seq(project_id),
        })
    
    else:
        logger.warning(f"[{project_id}] Unknown message type: {msg_type}")


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
            
            # Handle incoming message
            try:
                await handle_client_message(project_id, data)
            except Exception as e:
                logger.error(f"Error handling message from {project_id}: {e}", exc_info=True)
                # Send error response back to client
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to process message: {str(e)}",
                    "original_message": data,
                })
            
    except WebSocketDisconnect:
        await manager.disconnect(websocket, project_id)
        logger.info(f"Client disconnected from project {project_id}")
    except Exception as e:
        logger.error(f"WebSocket error for project {project_id}: {e}")
        await manager.disconnect(websocket, project_id)
