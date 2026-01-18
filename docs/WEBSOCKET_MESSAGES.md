# WebSocket Message Handling Documentation

## Overview

The WebSocket connection between frontend and backend now supports **bidirectional communication**. The backend can send events to the frontend, and the frontend can send messages/commands to the backend.

## Connection

**Endpoint:** `ws://localhost:8000/ws/projects/{project_id}`

**Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/projects/demo-project');
```

## Backend → Frontend Events

The backend sends events in this format:
```json
{
  "v": 1,
  "type": "EVENT",
  "project_id": "demo-project",
  "seq": 42,
  "ts": 1705532400000,
  "stage": {
    "id": "TRAIN",
    "index": 5,
    "status": "IN_PROGRESS"
  },
  "event": {
    "name": "TRAIN_PROGRESS",
    "payload": {
      "run_id": "run_123",
      "epoch": 10,
      "pct": 0.5
    }
  }
}
```

**Common Event Types:**
- `HELLO` - Initial connection acknowledgment
- `STAGE_STATUS` - Stage progression updates
- `TRAIN_PROGRESS` - Training progress updates
- `METRIC_SCALAR` - Training metrics (loss, accuracy, etc.)
- `LOG_LINE` - Log messages
- `EXPORT_READY` - Export bundle ready
- And 30+ other event types...

## Frontend → Backend Messages

The frontend can send messages to the backend to trigger actions.

### Message Format

```json
{
  "type": "message_type",
  "... additional fields ..."
}
```

### Supported Message Types

#### 1. Chat Messages

Send user chat messages or change requests.

```json
{
  "type": "chat",
  "text": "Please optimize for accuracy over speed"
}
```

**Backend Response:**
- Echoes the message as a `LOG_LINE` event
- In future: Routes to LangChain agent for processing

**Example:**
```javascript
wsClient.send(JSON.stringify({
  type: 'chat',
  text: 'Can you retrain with higher learning rate?'
}));
```

---

#### 2. Confirm Stage

Confirm the current stage and advance to the next stage.

```json
{
  "type": "confirm"
}
```

**Backend Response:**
- Calls `conductor.confirm(project_id)`
- Emits `STAGE_STATUS` events for stage transitions
- Updates stage from `WAITING_CONFIRMATION` to `COMPLETED`
- Advances to next stage with `IN_PROGRESS` status

**Example:**
```javascript
wsClient.send(JSON.stringify({
  type: 'confirm'
}));
```

---

#### 3. Commands

Execute specific commands on the backend.

```json
{
  "type": "command",
  "command": "command_name",
  "args": {
    "key": "value"
  }
}
```

**Available Commands:**

##### a) Restart Stage
```json
{
  "type": "command",
  "command": "restart_stage",
  "args": {
    "stage_id": "TRAIN"
  }
}
```

##### b) Cancel Run
```json
{
  "type": "command",
  "command": "cancel_run",
  "args": {}
}
```

**Example:**
```javascript
wsClient.send(JSON.stringify({
  type: 'command',
  command: 'restart_stage',
  args: { stage_id: 'PROFILE_DATA' }
}));
```

---

#### 4. Ping/Pong

Keepalive ping to check connection health.

```json
{
  "type": "ping",
  "ts": 1705532400000
}
```

**Backend Response:**
```json
{
  "type": "pong",
  "ts": 1705532400000,
  "server_ts": 123
}
```

**Example:**
```javascript
// Send ping every 30 seconds
setInterval(() => {
  wsClient.send(JSON.stringify({
    type: 'ping',
    ts: Date.now()
  }));
}, 30000);
```

---

## Error Handling

If the backend fails to process a message, it sends an error response:

```json
{
  "type": "error",
  "message": "Failed to process message: Invalid stage_id",
  "original_message": { ... }
}
```

## Frontend Integration Example

```typescript
import { createWebSocketClient } from '@/lib/ws';

const wsClient = createWebSocketClient({
  projectId: 'demo-project',
  baseUrl: 'ws://localhost:8000',
  
  onStatusChange: (status) => {
    console.log('Connection status:', status);
  },
  
  onEvent: (event) => {
    // Handle incoming events from backend
    console.log('Event:', event.event?.name, event.event?.payload);
  },
  
  onError: (err) => {
    console.error('WebSocket error:', err);
  }
});

// Send chat message
wsClient.send({
  type: 'chat',
  text: 'Please use Random Forest instead of XGBoost'
});

// Confirm stage
wsClient.send({
  type: 'confirm'
});

// Restart training
wsClient.send({
  type: 'command',
  command: 'restart_stage',
  args: { stage_id: 'TRAIN' }
});

// Cleanup
wsClient.close();
```

## Backend Implementation

The message handler is implemented in `backend/app/ws/router.py`:

```python
async def handle_client_message(project_id: str, data: Dict[str, Any]):
    msg_type = data.get("type", "unknown")
    
    if msg_type == "chat":
        # Handle chat messages
        # Echo to all clients + route to LangChain agent
        
    elif msg_type == "confirm":
        # Advance pipeline stage
        await conductor.confirm(project_id)
        
    elif msg_type == "command":
        # Execute specific commands
        command = data.get("command")
        args = data.get("args", {})
        
    elif msg_type == "ping":
        # Respond with pong
```

## Testing

Run the test script to verify WebSocket message handling:

```bash
cd backend
source .venv/bin/activate
python test_websocket_messages.py
```

Expected output:
```
✓ Connected to WebSocket
📨 Received HELLO: HELLO
📨 Received STAGE_STATUS: STAGE_STATUS
🧪 Test 1: Sending chat message...
✓ Received response: LOG_LINE
🧪 Test 2: Sending ping...
✓ Received pong
🧪 Test 3: Sending confirm...
✓ Received stage update: STAGE_STATUS
🧪 Test 4: Sending restart_stage command...
✓ Received command response: STAGE_STATUS
✅ All tests passed!
```

## Future Enhancements

- [ ] LangChain agent integration for chat messages
- [ ] Training run cancellation logic
- [ ] Custom command handlers
- [ ] Message authentication/authorization
- [ ] Rate limiting for messages
- [ ] Message queue for async processing

## See Also

- `backend/app/ws/router.py` - WebSocket endpoint and message handler
- `backend/app/ws/hub.py` - Connection manager
- `backend/app/events/bus.py` - Event bus for pub/sub
- `frontend/src/lib/ws.ts` - Frontend WebSocket client
- `docs/FRONTEND_BACKEND_CONTRACT.md` - Complete event specification
