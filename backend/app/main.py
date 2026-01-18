"""
AutoML Agentic Builder - Backend API
"""
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)
print(f"[Startup] Loaded environment from: {env_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="AutoML Agentic Builder",
    description="Backend API for the AutoML Agentic Builder",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from app.ws.router import router as ws_router
from app.api.test import router as test_router
from app.api.stages import router as stages_router
from app.api.assets import router as assets_router
from app.api.intent import router as intent_router
from app.api.data import router as data_router
from app.api.train import router as train_router
from app.api.export import router as export_router
from app.api.inference import router as inference_router
from app.api.predict import router as predict_router

# Include WebSocket router
app.include_router(ws_router)

# Include test router
app.include_router(test_router)

# Include stage state router
app.include_router(stages_router)

# Include assets router
app.include_router(assets_router)

# Include intent/model selection router
app.include_router(intent_router)

# Include data ingestion/profiling router
app.include_router(data_router)

# Include training router
app.include_router(train_router)

# Include inference router
app.include_router(inference_router)

# Include prediction router
app.include_router(predict_router, prefix="/api/projects", tags=["predictions"])

# Include report/export router
app.include_router(export_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "automl-backend"}


@app.get("/")
async def root():
    """Root endpoint with API overview."""
    return {
        "message": "AutoML Agentic Builder API",
        "version": "0.1.0",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "websocket": "/ws/projects/{project_id}"
        },
        "websocket_info": {
            "url": "ws://localhost:8000/ws/projects/{project_id}",
            "protocol": "WebSocket",
            "description": "Real-time bidirectional communication for ML pipeline events and commands",
            "documentation": "See /ws/docs for WebSocket API details"
        }
    }


@app.get("/ws/docs")
async def websocket_docs():
    """WebSocket API documentation (since WebSockets don't appear in OpenAPI/Swagger)."""
    return {
        "title": "WebSocket API Documentation",
        "endpoint": "/ws/projects/{project_id}",
        "description": "Real-time bidirectional communication for ML pipeline",
        "connection": {
            "url": "ws://localhost:8000/ws/projects/{project_id}",
            "example": "ws://localhost:8000/ws/projects/demo-project"
        },
        "server_to_client_events": [
            "HELLO - Initial connection acknowledgment",
            "STAGE_STATUS - Stage progression updates",
            "TRAIN_PROGRESS - Training progress updates",
            "METRIC_SCALAR - Training metrics (loss, accuracy)",
            "LOG_LINE - Log messages",
            "EXPORT_READY - Export bundle ready",
            "And 30+ other event types..."
        ],
        "client_to_server_messages": {
            "chat": {
                "description": "Send user chat messages or change requests",
                "example": {"type": "chat", "text": "Please optimize for accuracy"}
            },
            "confirm": {
                "description": "Confirm current stage and advance pipeline",
                "example": {"type": "confirm"}
            },
            "command": {
                "description": "Execute specific commands",
                "example": {
                    "type": "command",
                    "command": "restart_stage",
                    "args": {"stage_id": "TRAIN"}
                }
            },
            "ping": {
                "description": "Keepalive ping",
                "example": {"type": "ping", "ts": 1705532400000}
            }
        },
        "full_documentation": "/docs/WEBSOCKET_MESSAGES.md in repository"
    }
