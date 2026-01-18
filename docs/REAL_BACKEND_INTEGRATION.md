# Real Backend Integration - NO MOCK DATA

## What Changed

The frontend has been completely rewired to use **REAL backend data** through WebSocket and REST APIs. All mock/fake data has been disabled.

## New Components

### 1. `useBackendPipeline` Hook
**File:** `frontend/src/hooks/useBackendPipeline.ts`

**Purpose:** React hook that manages real backend communication

**Features:**
- Connects to WebSocket for real-time events
- Calls REST APIs for pipeline actions
- Tracks pipeline state and stages
- **NO FALLBACKS** - if backend fails, it shows errors clearly

**Usage:**
```typescript
const {
  isRunning,
  currentStage,
  error,
  events,
  stages,
  connectionStatus,
  startPipeline,
  uploadDataset,
  confirmStage,
  sendChatMessage,
} = useBackendPipeline({
  projectId: 'demo-project',
  onStageChange: (stage) => console.log('Stage:', stage),
  onError: (error) => console.error('Error:', error),
  onComplete: () => console.log('Complete!'),
});
```

### 2. `RealBackendLoader` Component
**File:** `frontend/src/components/center/loader/RealBackendLoader.tsx`

**Purpose:** UI component that displays real backend pipeline status

**Features:**
- ✅ Shows connection status (connecting, open, error, closed)
- ✅ Displays all WebSocket events in real-time
- ✅ Shows current stage and status
- ✅ Allows stage confirmation and chat messages
- ❌ **NO MOCK DATA** - shows clear errors if backend is down

**Error States:**
1. **Backend Not Running** - Shows instructions to start backend
2. **Connection Error** - Shows WebSocket connection details
3. **Pipeline Error** - Shows exact error message from backend

## Integration Points

### PreviewPane Updated
**File:** `frontend/src/components/center/PreviewPane.tsx`

**Before:**
```tsx
import TrainingLoaderV2 from './preview/TrainingLoaderV2';  // Mock data

<TrainingLoaderV2 
  onComplete={handleComplete} 
  updateFileContent={updateFileContent} 
/>
```

**After:**
```tsx
import RealBackendLoader from './loader/RealBackendLoader';  // Real backend

<RealBackendLoader 
  onComplete={handleComplete} 
  updateFileContent={updateFileContent} 
/>
```

## Backend Requirements

The frontend **REQUIRES** the following backend to be running:

### 1. Start Backend Server
```bash
cd /Users/johndoe/Documents/VsCode_Files/AutoML/backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Required Backend Endpoints

#### WebSocket
- `ws://localhost:8000/ws/projects/{project_id}` - Real-time events

#### REST APIs
- `POST /api/projects/{project_id}/parse` - Parse user intent
- `POST /api/projects/{project_id}/upload` - Upload dataset
- `POST /api/projects/{project_id}/confirm` - Confirm stage
- `GET /api/projects/{project_id}/state` - Get project state

### 3. Required Environment Variables
```bash
# backend/.env
GEMINI_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here  # For chat agent
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

## What Happens If Backend Is Down

### ❌ NO MORE FAKE DATA

The app will show **clear error messages** instead of falling back to mocks:

**1. WebSocket Connection Failed:**
```
Backend Connection Failed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cannot connect to the backend WebSocket server. 
The backend may not be running.

Expected WebSocket URL:
ws://localhost:8000/ws/projects/demo-project

To fix this:
1. Open terminal in backend directory
2. Activate virtual environment: source .venv/bin/activate
3. Start server: uvicorn app.main:app --reload
4. Refresh this page

[Retry Connection]
```

**2. API Call Failed:**
```
Pipeline Error
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An error occurred while running the ML pipeline:

Failed to fetch: http://localhost:8000/api/projects/demo-project/parse
Network error - backend not responding

[Retry] [Dismiss]
```

## Event Flow

### Frontend → Backend (User Actions)
```
User clicks "Start Pipeline"
  ↓
Frontend calls: POST /api/projects/demo-project/parse
  ↓
Backend PromptParserAgent processes
  ↓
Backend emits: PROMPT_PARSED event via WebSocket
  ↓
Frontend receives event and updates UI
```

### Backend → Frontend (Events)
```
Backend training starts
  ↓
WebSocket emits: TRAIN_RUN_STARTED
  ↓
WebSocket emits: TRAIN_PROGRESS (multiple times)
  ↓
WebSocket emits: METRIC_SCALAR (multiple times)
  ↓
WebSocket emits: TRAIN_RUN_FINISHED
  ↓
Frontend displays all events in real-time
```

## Testing

### 1. Start Backend
```bash
cd /Users/johndoe/Documents/VsCode_Files/AutoML/backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

### 2. Start Frontend
```bash
cd /Users/johndoe/Documents/VsCode_Files/AutoML/frontend
npm run dev
```

### 3. Open Browser
```
http://localhost:5173
```

### 4. Expected Behavior

#### ✅ Backend Running
- Green "Live" indicator in top right
- "Connected to ws://localhost:8000" message
- Can click "Start Pipeline" button
- Events appear in real-time as backend processes them

#### ❌ Backend Not Running
- Red error message: "Backend Connection Failed"
- Instructions on how to start backend
- NO fake data shown

## Files Changed

### New Files
1. `frontend/src/hooks/useBackendPipeline.ts` - Backend integration hook
2. `frontend/src/components/center/loader/RealBackendLoader.tsx` - Real backend UI
3. `docs/WEBSOCKET_MESSAGES.md` - WebSocket API documentation

### Modified Files
1. `frontend/src/components/center/PreviewPane.tsx` - Use RealBackendLoader instead of TrainingLoaderV2
2. `backend/app/ws/router.py` - Added client message handling
3. `backend/app/orchestrator/conductor.py` - Added _get_current_stage_id helper
4. `backend/app/main.py` - Added /ws/docs endpoint
5. `backend/.env.example` - Added OPENROUTER_API_KEY documentation

## Next Steps

### Required
1. **Start backend server** - Frontend will NOT work without it
2. **Add OpenRouter API key** - For chat agent functionality
3. **Implement remaining backend agents:**
   - PromptParserAgent (exists)
   - ModelSelectorAgent (exists)
   - DataProfiler (exists)
   - PreprocessAgent (exists)
   - TrainingRunner (exists)
   - Need to wire them all together in conductor

### Optional
- Add loading states for API calls
- Add retry logic for failed requests
- Add offline mode detection
- Add request/response caching

## Debugging

### Check WebSocket Connection
```bash
# In browser console:
ws = new WebSocket('ws://localhost:8000/ws/projects/demo-project');
ws.onmessage = (e) => console.log('Event:', JSON.parse(e.data));
```

### Check REST API
```bash
curl http://localhost:8000/
curl http://localhost:8000/ws/docs
curl http://localhost:8000/health
```

### Check Backend Logs
```bash
# Backend terminal will show:
WebSocket connected for project demo-project
Publishing event HELLO to 1 subscribers
Publishing event STAGE_STATUS to 1 subscribers
```

## Important Notes

1. **NO MOCK DATA** - App will fail clearly if backend is down
2. **Real WebSocket Events** - All events come from backend
3. **Real REST APIs** - All actions call backend endpoints
4. **Clear Errors** - Shows exactly what went wrong
5. **No Silent Failures** - Every error is displayed to user

This ensures you know immediately if something is wrong with the backend integration!
