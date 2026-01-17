from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict

router = APIRouter()

# In-memory storage for chat history
chat_history: List[Dict[str, str]] = []

@router.post("/chat/send")
async def send_message(message: str):
    chat_history.append({"role": "user", "content": message})
    # Here you would typically process the message and generate a response
    response = "This is a placeholder response."
    chat_history.append({"role": "assistant", "content": response})
    return {"response": response}

@router.get("/chat/history")
async def get_chat_history():
    return chat_history

@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await send_message(data)
            await websocket.send_text(chat_history[-1]["content"])
    except WebSocketDisconnect:
        pass