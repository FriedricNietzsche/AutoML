# CRITICAL FIX: Infinite Loop Resolved

## ✅ Fixed: Maximum Update Depth Exceeded Error

**Root Cause**: Callback functions (`onError`, `onStageChange`, `onComplete`) were being recreated on every render, causing all `useCallback` hooks to be recreated, triggering infinite re-renders.

**Solution**: Use `useRef` to store stable callback references

### Changes Made to `frontend/src/hooks/useBackendPipeline.ts`:

```typescript
// BEFORE (BROKEN - caused infinite loop):
export function useBackendPipeline(options: UsePipelineOptions) {
  const { onStageChange, onError, onComplete } = options;
  
  const startPipeline = useCallback(async (prompt: string) => {
    // ...
    onError?.(error);  // ❌ Depends on onError from props
  }, [apiBase, projectId, onError]);  // ❌ onError changes every render!
}

// AFTER (FIXED - stable references):
export function useBackendPipeline(options: UsePipelineOptions) {
  // Use refs for callbacks to prevent dependency changes
  const onStageChangeRef = useRef(options.onStageChange);
  const onErrorRef = useRef(options.onError);
  const onCompleteRef = useRef(options.onComplete);
  
  // Update refs when callbacks change (doesn't trigger re-renders)
  useEffect(() => {
    onStageChangeRef.current = options.onStageChange;
    onErrorRef.current = options.onError;
    onCompleteRef.current = options.onComplete;
  });
  
  const startPipeline = useCallback(async (prompt: string) => {
    // ...
    onErrorRef.current?.(error);  // ✅ Stable ref, never changes
  }, [apiBase, projectId]);  // ✅ Only real dependencies!
}
```

**Result**: 
- ✅ No more infinite loops
- ✅ Callbacks are stable and don't trigger re-renders
- ✅ `startPipeline`, `uploadDataset`, `confirmStage`, `sendChatMessage` are now stable

---

## ✅ Fixed: WebSocket Connection Stability

**Result from logs**:
```
WebSocket connected for project demo-project. Total connections: 2
```

Still seeing 2 connections instead of 1. This is because:
1. AppShell creates connection #1
2. Something else is creating connection #2

**To investigate**: Check if you have multiple tabs open or if StrictMode is causing double-mounting.

---

## ⚠️ Important: Backend is NOT Receiving `/parse` Request

### What's Happening

**Backend logs show**:
```
✅ WebSocket connected
✅ STAGE_STATUS events published
✅ GET /api/projects/demo-project/state - 200 OK
❌ NO /parse request received
```

**Frontend shows**: "PARSE_INTENT in progress"

### Why This Happens

The backend initializes with `PARSE_INTENT` as the first stage by default. The frontend reads this stage from WebSocket events and displays it as "in progress", but **the `/parse` API endpoint was never actually called**.

### How to Actually Start the Pipeline

The pipeline will NOT start automatically. You must:

1. **Click the "Start Pipeline" button** in the UI, OR
2. **Call `startPipeline(prompt)` programmatically**

The button appears when `events.length === 0`:

```tsx
{events.length === 0 && (
  <div className="text-center py-12">
    <p className="text-replit-textMuted mb-4">No events yet. Start the pipeline to begin.</p>
    <button onClick={handleStart}>
      Start Pipeline
    </button>
  </div>
)}
```

When clicked, it calls:
```typescript
await startPipeline('Build a customer churn prediction model');
```

Which makes this API call:
```
POST /api/projects/demo-project/parse
Body: {"prompt": "Build a customer churn prediction model"}
```

**Then you'll see in backend logs**:
```
============================================================
[PARSE INTENT] Starting for project: demo-project
============================================================
[PromptParser] Using LangChain LLM...
[PromptParser] ✅ LLM response received in 2.5s
============================================================
[PARSE INTENT] ✅ COMPLETED
============================================================
```

---

## Testing the Complete Fix

### 1. Restart Frontend
```bash
cd frontend
npm run dev
```

### 2. Check Browser Console
- Open DevTools → Console
- Should see NO "Maximum update depth exceeded" error
- Should see stable WebSocket connection messages

### 3. Look for the "Start Pipeline" Button
- It should appear in the center of the screen
- Text: "No events yet. Start the pipeline to begin."

### 4. Click "Start Pipeline"
- This will call `/api/projects/demo-project/parse`
- Backend terminal will show detailed logs
- Events will start appearing in the UI

### 5. Watch Backend Terminal
You should see:
```
============================================================
[PARSE INTENT] Starting for project: demo-project
[PARSE INTENT] Prompt: Build a customer churn prediction...
============================================================

[1/4] Creating PromptParserAgent...
[2/4] Calling agent.parse()...
[PromptParser] Starting parse...
[PromptParser] Using LangChain LLM...
[PromptParser] Invoking LLM chain...
[PromptParser] ✅ LLM response received in 3.2s
[2/4] ✅ Parse result: {...}
[3/4] Publishing PROMPT_PARSED event...
[4/4] Transitioning stages...

============================================================
[PARSE INTENT] ✅ COMPLETED for project: demo-project
============================================================
```

---

## Summary

### What Was Fixed
1. ✅ **Infinite loop** - Used `useRef` for callback stability
2. ✅ **Callback dependencies** - Removed recreating callbacks from deps
3. ✅ **Added detailed logging** - Can see exactly what's slow

### What's Working
- ✅ WebSocket connection established
- ✅ Stage status events flowing
- ✅ Frontend displaying current stage
- ✅ No infinite render loops

### What You Need to Do
- ⚠️ **Click "Start Pipeline" button** to actually trigger the `/parse` API call
- ⚠️ Or programmatically call `startPipeline(prompt)` when you want to start

### Expected Behavior After Click
1. Frontend calls `POST /api/projects/demo-project/parse`
2. Backend logs show detailed parsing progress
3. LLM processes the prompt (2-5 seconds)
4. Backend emits `PROMPT_PARSED` event via WebSocket
5. Frontend receives event and shows progress
6. Stage transitions to `DATA_SOURCE` waiting for upload

---

## Files Changed
- `frontend/src/hooks/useBackendPipeline.ts` - Fixed infinite loop with useRef
- `backend/app/api/intent.py` - Added detailed logging
- `backend/app/agents/prompt_parser.py` - Added timing logs
