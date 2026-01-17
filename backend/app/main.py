from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import projects, chat, stages, assets
from app.ws.hub import websocket_router

app = FastAPI()

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(stages.router, prefix="/api/stages", tags=["stages"])
app.include_router(assets.router, prefix="/api/assets", tags=["assets"])

# Include WebSocket router
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])

@app.get("/")
async def root():
    return {"message": "Welcome to the AutoML Agentic Builder API!"}