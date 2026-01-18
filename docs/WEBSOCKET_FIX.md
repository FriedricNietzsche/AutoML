# WebSocket Connection Stability Fix

## Problem
Multiple WebSocket connections were being created simultaneously, causing immediate disconnects:
- Backend logs showed 4 connections created then immediately closed
- Frontend showed "WebSocket connection failed" despite backend accepting connections
- Connection churn caused by React useEffect dependency loops

## Root Causes

### 1. **AppShell useEffect Dependency Loop**
```tsx
// BEFORE (BROKEN):
useEffect(() => {
  connectProject({ projectId, wsBase });
  hydrate();
}, [connectProject, hydrate, projectId, wsBase]);
// ❌ connectProject and hydrate are recreated on every render
// ❌ This causes reconnection on EVERY render
```

**Fix**: Removed `connectProject` and `hydrate` from dependencies
```tsx
// AFTER (FIXED):
useEffect(() => {
  connectProject({ projectId, wsBase });
  hydrate();
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [projectId, wsBase]);
// ✅ Only reconnect when projectId or wsBase actually changes
```

### 2. **Missing Connection Guards in projectStore**
No guards prevented duplicate connections to the same project.

**Fix**: Added connection state guards
```typescript
connect: (opts) => {
  const currentState = get();
  
  // Guard 1: Don't reconnect if already connected
  if (
    currentState.connectionStatus === 'open' &&
    currentState.projectId === projectId &&
    currentState.wsBase === wsBase &&
    currentState.wsClient
  ) {
    console.log('Already connected - skipping reconnect');
    return;
  }

  // Guard 2: Don't create new connection if currently connecting
  if (
    currentState.connectionStatus === 'connecting' &&
    currentState.projectId === projectId &&
    currentState.wsBase === wsBase
  ) {
    console.log('Already connecting - skipping duplicate');
    return;
  }

  // Rest of connection logic...
}
```

### 3. **useBackendPipeline Hook Documentation**
Added clear documentation that hook uses **EXISTING** connection, doesn't create new one.

```typescript
/**
 * Real backend pipeline integration hook
 * Uses the EXISTING WebSocket connection from projectStore
 * NO duplicate connections - reuses the global connection managed by AppShell
 */
```

## Files Changed

1. **frontend/src/components/shell/AppShell.tsx**
   - Removed `connectProject` and `hydrate` from useEffect dependencies
   - Added eslint-disable comment with explanation
   - Connection only triggered when `projectId` or `wsBase` changes

2. **frontend/src/store/projectStore.ts**
   - Added guard: skip reconnect if already connected to same project
   - Added guard: skip duplicate if already connecting to same project
   - Added console logs for debugging connection attempts

3. **frontend/src/hooks/useBackendPipeline.ts**
   - Updated documentation to clarify it uses EXISTING connection
   - Removed unused `wsClient` from destructuring (uses `getState()` in sendChatMessage)

## Expected Behavior After Fix

### Single Connection Per Project
- ✅ One WebSocket connection created on mount
- ✅ Connection stays open (no immediate disconnects)
- ✅ No duplicate connections created on re-renders
- ✅ Reconnect ONLY when projectId or wsBase changes

### Backend Logs
```
WebSocket connected for project demo-project. Total connections: 1
Subscriber added for project demo-project. Total: 1
connection open
[Connection stays open - no immediate disconnects]
```

### Frontend Logs
```
Already connected to demo-project - skipping reconnect
[No duplicate connection attempts]
[connectionStatus: 'open']
```

## Testing

1. **Start backend:**
   ```bash
   cd backend
   source .venv/bin/activate
   uvicorn app.main:app --reload
   ```

2. **Start frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Verify in browser console:**
   - Check for "Already connected" messages (means guards working)
   - Should see NO multiple connection attempts
   - WebSocket tab in DevTools should show 1 connection

4. **Verify in backend logs:**
   - Should see "Total connections: 1"
   - NO immediate "WebSocket disconnected" messages
   - Connection should stay stable

## Architecture

```
┌─────────────────────────────────────────────────┐
│ AppShell (Shell Component)                      │
│ - Mounts once                                   │
│ - Calls projectStore.connect() ONCE             │
│ - Manages global WebSocket connection           │
└─────────────────────┬───────────────────────────┘
                      │
                      │ creates & manages
                      ▼
┌─────────────────────────────────────────────────┐
│ projectStore (Zustand)                          │
│ - connect() with guards                         │
│ - Single wsClient instance                      │
│ - Broadcasts to all subscribers                 │
└─────────────────────┬───────────────────────────┘
                      │
                      │ reuses connection
                      ▼
┌─────────────────────────────────────────────────┐
│ useBackendPipeline Hook                         │
│ - NO connection creation                        │
│ - Reads from projectStore                       │
│ - Uses existing wsClient                        │
└─────────────────────────────────────────────────┘
                      │
                      │ uses hook
                      ▼
┌─────────────────────────────────────────────────┐
│ RealBackendLoader Component                     │
│ - Displays events from store                    │
│ - Sends messages via store's wsClient           │
│ - NO connection management                      │
└─────────────────────────────────────────────────┘
```

## Key Principles

1. **Single Source of Truth**: AppShell manages WebSocket connection
2. **No Duplicate Connections**: Guards prevent multiple simultaneous connections
3. **Stable Dependencies**: Only reconnect on actual data changes
4. **Shared Connection**: All components use the same wsClient from store
5. **Clear Logging**: Console messages show when guards activate

## Next Steps

After applying this fix:
1. Restart both backend and frontend
2. Verify single connection in logs
3. Test pipeline flow (parse → upload → train)
4. Confirm events stream correctly
5. Test stage confirmations and chat messages

## Related Documentation
- [WEBSOCKET_MESSAGES.md](./WEBSOCKET_MESSAGES.md) - WebSocket API reference
- [REAL_BACKEND_INTEGRATION.md](./REAL_BACKEND_INTEGRATION.md) - Integration guide
- [FRONTEND_BACKEND_CONTRACT.md](./FRONTEND_BACKEND_CONTRACT.md) - Event contracts
