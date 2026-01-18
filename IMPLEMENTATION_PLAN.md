# Implementation Plan: Restore Original Workflow

## Current Problem
- RealBackendLoader hardcodes the prompt instead of reading from BuildSession
- No clear "Start Pipeline" button flow
- Confirmation flow not properly connected

## Original Workflow (Target)
```
┌─────────────────────────────────────────────────────────────────┐
│ 1. HomePage: User enters prompt + optional dataset              │
│    → Click "Continue"                                            │
│    → Creates BuildSession { goalPrompt, datasetLink }            │
│    → Navigate to /workspace                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ 2. Workspace: Shows RealBackendLoader                            │
│    → Displays session.goalPrompt                                 │
│    → Shows "Start Pipeline" button                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ 3. User clicks "Start Pipeline"                                  │
│    → Calls startPipeline(session.goalPrompt)                     │
│    → POST /api/projects/demo-project/parse                       │
│    → Backend processes with LLM (2-5s)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ 4. Backend emits events via WebSocket                            │
│    → PROMPT_PARSED                                               │
│    → STAGE_STATUS (PARSE_INTENT → COMPLETED)                     │
│    → STAGE_STATUS (DATA_SOURCE → WAITING_CONFIRMATION)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ 5. Frontend shows "Confirm & Continue" button                    │
│    → User clicks                                                 │
│    → Calls confirmStage()                                        │
│    → POST /api/projects/demo-project/confirm                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ 6. Repeat for each stage:                                        │
│    DATA_SOURCE → DATA_CLEAN → FEATURE_ENG → MODEL_SELECT →      │
│    TRAIN → EVALUATE → EXPORT                                     │
│                                                                   │
│    Each stage:                                                   │
│    - Shows progress events                                       │
│    - Waits for user confirmation                                 │
│    - Advances to next stage                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│ 7. Final Stage (EXPORT) completes                                │
│    → onComplete() called                                         │
│    → Shows API docs / results                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Tasks

### ✅ TASK 1: Pass BuildSession to RealBackendLoader
**File**: `frontend/src/components/center/PreviewPane.tsx`
- Pass `session` prop to RealBackendLoader
- RealBackendLoader reads `session.goalPrompt` instead of hardcoding

### ✅ TASK 2: Update RealBackendLoader to Accept Session
**File**: `frontend/src/components/center/loader/RealBackendLoader.tsx`
- Add `session: BuildSession` to props
- Remove hardcoded prompt
- Show session.goalPrompt in UI before starting
- Use session.goalPrompt when calling startPipeline()

### ✅ TASK 3: Create Initial Prompt Display State
**File**: `frontend/src/components/center/loader/RealBackendLoader.tsx`
- Show prompt review screen BEFORE pipeline starts
- Display: "Your Goal: {session.goalPrompt}"
- Display: "Dataset: {session.datasetLink}" if provided
- Big "Start Pipeline" button
- Only after click → call startPipeline()

### ✅ TASK 4: Implement Stage-by-Stage Confirmation
**File**: `frontend/src/components/center/loader/RealBackendLoader.tsx`
- After each stage completes → show "Confirm & Continue"
- Button calls confirmStage()
- Backend advances to next stage
- Frontend waits for STAGE_STATUS event
- Repeat until EXPORT

### ✅ TASK 5: Update AppShell to Pass Session
**File**: `frontend/src/components/shell/AppShell.tsx`
- Get current BuildSession from storage
- Pass to PreviewPane
- PreviewPane passes to RealBackendLoader

### ✅ TASK 6: Handle Dataset Upload Flow
**File**: `frontend/src/components/center/loader/RealBackendLoader.tsx`
- When DATA_SOURCE stage is WAITING_CONFIRMATION
- Show file upload UI or dataset link display
- Upload file using uploadDataset()
- OR show "Using dataset: {session.datasetLink}"

### ✅ TASK 7: Test Complete Flow
- Start from HomePage with prompt
- Navigate to workspace
- See prompt review
- Click Start
- Watch parse happen
- Confirm each stage
- Reach EXPORT

## Files to Modify

1. `frontend/src/components/shell/AppShell.tsx`
   - Import getCurrentSession
   - Pass session to PreviewPane

2. `frontend/src/components/center/PreviewPane.tsx`
   - Accept session prop
   - Pass to RealBackendLoader

3. `frontend/src/components/center/loader/RealBackendLoader.tsx`
   - Accept session prop
   - Add "prompt review" initial state
   - Use session.goalPrompt for API calls
   - Show dataset info from session.datasetLink
   - Implement confirmation buttons properly

4. `frontend/src/hooks/useBackendPipeline.ts`
   - Already fixed (no changes needed)

## Success Criteria
- ✅ User enters prompt on HomePage
- ✅ Session created and stored
- ✅ Workspace shows prompt review
- ✅ "Start Pipeline" button visible
- ✅ Click → calls /parse with correct prompt
- ✅ Backend processes and streams events
- ✅ Each stage waits for confirmation
- ✅ All stages complete to EXPORT
- ✅ No hardcoded prompts
- ✅ No infinite loops
- ✅ WebSocket connection stable (1 connection)
