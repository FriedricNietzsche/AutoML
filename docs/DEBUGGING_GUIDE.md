# Debugging Guide

## Fixed Issues

### 1. ✅ Maximum Update Depth Exceeded Error

**Problem**: Infinite loop in `useBackendPipeline` hook causing React error
```
Maximum update depth exceeded. This can happen when a component calls setState 
inside useEffect, but useEffect either doesn't have a dependency array, or one 
of the dependencies changes on every render.
```

**Root Cause**: `state.currentStage` was in the useEffect dependency array:
```tsx
// BROKEN:
useEffect(() => {
  if (currentStageId && currentStageId !== state.currentStage) {
    setState(prev => ({ ...prev, currentStage: currentStageId }));
  }
}, [currentStageId, state.currentStage]); // ❌ state.currentStage causes infinite loop
```

**Fix**: Removed `state.currentStage` from dependencies
```tsx
// FIXED:
useEffect(() => {
  if (currentStageId && currentStageId !== state.currentStage) {
    setState(prev => ({ ...prev, currentStage: currentStageId }));
  }
  // DO NOT include state.currentStage in deps - causes infinite loop
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [currentStageId, stages, onStageChange, onComplete]);
```

**File Changed**: `frontend/src/hooks/useBackendPipeline.ts` line 47

---

### 2. ✅ Parse Intent Performance Logging

**Problem**: Parse intent taking too long with no visibility into what's happening

**Solution**: Added detailed logging throughout the parse pipeline

**Logs Added**:

#### API Endpoint (`backend/app/api/intent.py`)
```
============================================================
[PARSE INTENT] Starting for project: demo-project
[PARSE INTENT] Prompt: Build a cat vs dog classifier...
============================================================

[1/4] Creating PromptParserAgent...
[2/4] Calling agent.parse()...
[2/4] ✅ Parse result: {task_type: 'classification', ...}
[3/4] Publishing PROMPT_PARSED event...
[3/4] ✅ Event published
[4/4] Transitioning stages...
[4/4] ✅ Stages transitioned

============================================================
[PARSE INTENT] ✅ COMPLETED for project: demo-project
============================================================
```

#### Parser Agent (`backend/app/agents/prompt_parser.py`)
```
[PromptParser] Starting parse...
[PromptParser] Prompt length: 150 chars
[PromptParser] Using LangChain LLM...
[PromptParser] Invoking LLM chain...
[PromptParser] ✅ LLM response received in 2.34s
[PromptParser] Validating response...
[PromptParser] ✅ Complete in 2.45s (LangChain)
```

**Benefits**:
- ✅ See exactly which step is slow (LLM call vs validation vs DB)
- ✅ Track LLM response time
- ✅ Identify if LangChain path fails and fallback is used
- ✅ Detect API key issues early

**Files Changed**:
- `backend/app/api/intent.py` - Added step-by-step logging
- `backend/app/agents/prompt_parser.py` - Added timing and path logging

---

## How to Use Logging

### Backend Terminal
Watch the Python terminal where you ran:
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

You'll see detailed logs like:
```
[PARSE INTENT] Starting for project: demo-project
[PromptParser] Using LangChain LLM...
[PromptParser] Invoking LLM chain...
[PromptParser] ✅ LLM response received in 3.21s
[PARSE INTENT] ✅ COMPLETED
```

### Performance Diagnosis

**If parsing is slow:**

1. **Check LLM response time**:
   ```
   [PromptParser] ✅ LLM response received in 15.3s
   ```
   - If > 10s: LLM API is slow (OpenRouter/Gemini)
   - If < 2s: LLM is fast, bottleneck elsewhere

2. **Check which path is used**:
   ```
   [PromptParser] Using LangChain LLM...        ← Preferred path
   [PromptParser] Using legacy OpenRouter...    ← Fallback 1
   [PromptParser] Using heuristic fallback...   ← Fallback 2
   ```

3. **Check for errors**:
   ```
   [PromptParser] ⚠️  LangChain failed: API key invalid
   [PromptParser] ❌ Validation error: task_type missing
   ```

### Common Issues

| Log Message | Meaning | Fix |
|------------|---------|-----|
| `LLM response received in 20s+` | API is slow/overloaded | Wait or switch LLM provider |
| `LangChain failed: API key` | GEMINI_API_KEY invalid | Check `.env` file |
| `Using heuristic fallback` | Both LLMs failed | Check API keys, internet connection |
| `Validation error` | LLM returned bad JSON | LLM prompt may need tuning |

---

## Testing the Fix

1. **Restart backend** (to load new logging):
   ```bash
   cd backend
   source .venv/bin/activate
   uvicorn app.main:app --reload
   ```

2. **Restart frontend** (to load infinite loop fix):
   ```bash
   cd frontend
   npm run dev
   ```

3. **Test parse intent**:
   - Enter prompt: "Build a cat vs dog classifier"
   - Click "Continue"
   - Watch backend terminal for detailed logs

4. **Verify no infinite loop**:
   - Open browser DevTools → Console
   - Should NOT see "Maximum update depth exceeded"
   - Should see stable WebSocket connection

---

## Performance Expectations

### Normal Timing
```
[1/4] Create agent:      < 0.1s
[2/4] LLM call:          2-5s  (depends on API)
[3/4] Publish event:     < 0.1s
[4/4] Transition stages: < 0.1s
──────────────────────────────
Total:                   2-5s
```

### Slow Timing (Investigate)
```
[1/4] Create agent:      < 0.1s
[2/4] LLM call:          15-30s  ⚠️  API slow!
[3/4] Publish event:     < 0.1s
[4/4] Transition stages: < 0.1s
──────────────────────────────
Total:                   15-30s  ⚠️
```

---

## Related Documentation
- [WEBSOCKET_FIX.md](./WEBSOCKET_FIX.md) - Connection stability fixes
- [REAL_BACKEND_INTEGRATION.md](./REAL_BACKEND_INTEGRATION.md) - Integration guide
- [WEBSOCKET_MESSAGES.md](./WEBSOCKET_MESSAGES.md) - WebSocket API
