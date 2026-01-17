"""
AutoML Agentic Builder - Backend API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

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

# Include WebSocket router
app.include_router(ws_router)

# Include test router
app.include_router(test_router)

# Include stage state router
app.include_router(stages_router)

# Include assets router
app.include_router(assets_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "automl-backend"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AutoML Agentic Builder API",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/projects/{project_id}"
    }
