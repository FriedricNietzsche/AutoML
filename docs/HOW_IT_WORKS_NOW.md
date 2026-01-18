# How The System Works Now (with issues identified)

## Current Flow

### 1. **HomePage → Workspace Navigation**
```
User enters prompt: "Build a cat classifier"
  ↓
Creates BuildSession:
  {
    goalPrompt: "Build a cat classifier",
    datasetLinks: [...],
    status: "building"
  }
  ↓
Navigate to /workspace
  ↓
RealBackendLoader mounts
```

### 2. **WebSocket Connection (Automatic)**
```
AppShell useEffect runs:
  ↓
Calls: projectStore.connect({ projectId: "demo-project" })
  ↓
WebSocket connects to: ws://localhost:8000/ws/projects/demo-project
  ↓
Backend sends initial events:
  - HELLO (connection established)
  - STAGE_STATUS (current stage: PARSE_INTENT, status: PENDING)
```

### 3. **Prompt Review Screen**
```
RealBackendLoader checks:
  - pipelineStarted? NO
  - hasPipelineEvents? NO (only HELLO & STAGE_STATUS)
  ↓
Shows Prompt Review Screen:
  ✨ Ready to Start
  📝 Your Goal: "Build a cat classifier"
  🚀 [Start Pipeline] button
```

### 4. **User Clicks "Start Pipeline"**
```
handleStart() called:
  ↓
Sets: pipelineStarted = true
  ↓
Calls: startPipeline(session.goalPrompt)
  ↓
Makes HTTP Request:
  POST /api/projects/demo-project/parse
  Body: {"prompt": "Build a cat classifier"}
```

### 5. **Backend Processes Parse Intent**
```
Backend receives /parse request:
  ↓
Creates PromptParserAgent
  ↓
Calls LLM (Gemini/OpenRouter):
  - Analyzes prompt
  - Extracts: task_type, target, dataset_hint, constraints
  - Takes 2-5 seconds
  ↓
Returns parsed payload:
  {
    "task_type": "classification",
    "target": "cat vs dog",
    "dataset_hint": "cat and dog images",
    "constraints": {}
  }
  ↓
Publishes WebSocket Event:
  EventType.PROMPT_PARSED (payload: parsed data)
  ↓
Updates stage status:
  PARSE_INTENT → COMPLETED
  DATA_SOURCE → WAITING_CONFIRMATION
```

### 6. **Frontend Receives Events**
```
WebSocket receives events:
  ↓
Events stored in projectStore
  ↓
useBackendPipeline hook updates:
  - events array grows
  - currentStage changes to DATA_SOURCE
  - stages[DATA_SOURCE].status = WAITING_CONFIRMATION
  ↓
RealBackendLoader re-renders:
  - Shows event stream
  - Shows "Confirm & Continue" button
```

### 7. **User Clicks "Confirm & Continue"**
```
confirmStage() called:
  ↓
Makes HTTP Request:
  POST /api/projects/demo-project/confirm
  ↓
Backend conductor.confirm():
  ❌ PROBLEM: Doesn't check if stage is WAITING_CONFIRMATION
  ✅ Marks current stage COMPLETED
  ✅ Advances to next stage (DATA_CLEAN)
  ✅ Sets next stage to IN_PROGRESS
  ↓
Publishes events via WebSocket
  ↓
Frontend updates and cycle repeats
```

---

## 🐛 **IDENTIFIED ISSUES**

### Issue 1: Confirm Can Be Spammed
**Problem**: `conductor.confirm()` doesn't validate that the current stage is `WAITING_CONFIRMATION`

**Current Code**:
```python
async def confirm(self, project_id: str) -> Dict[str, Any]:
    # ❌ No check if stages[current_stage]["status"] == WAITING_CONFIRMATION
    stages[current_stage]["status"] = StageStatus.COMPLETED
    # Advances to next stage blindly
```

**Result**: User can spam "Confirm" button and skip through ALL stages instantly without any actual work happening.

**Fix Needed**: Add validation:
```python
async def confirm(self, project_id: str) -> Dict[str, Any]:
    current_stage: StageID = state["current_stage"]
    
    # ✅ Add this check
    if stages[current_stage]["status"] != StageStatus.WAITING_CONFIRMATION:
        raise HTTPException(
            status_code=400, 
            detail=f"Stage {current_stage.value} is not waiting for confirmation"
        )
    
    # Then proceed with confirmation logic
```

---

### Issue 2: Stages Don't Do Actual Work
**Problem**: Most stages just transition immediately without calling agents or doing real work.

**Current Behavior**:
- ✅ **PARSE_INTENT**: Actually calls PromptParserAgent (LLM processes)
- ❌ **DATA_SOURCE**: No agent called, just waits for confirm
- ❌ **DATA_CLEAN**: No agent called, just waits for confirm
- ❌ **FEATURE_ENG**: No agent called, just waits for confirm
- ❌ **MODEL_SELECT**: Has `ModelSelectorAgent` but not wired to /confirm
- ❌ **TRAIN**: No agent called, just waits for confirm
- ❌ **EVALUATE**: No agent called, just waits for confirm
- ❌ **EXPORT**: No agent called, just waits for confirm

**What Should Happen**:
Each stage should trigger actual work BEFORE going to WAITING_CONFIRMATION:

```python
# Example for DATA_CLEAN stage:
async def confirm(self, project_id: str):
    next_stage = advance_to_next()
    
    # When advancing to DATA_CLEAN, trigger agent
    if next_stage == StageID.DATA_CLEAN:
        # Call DataCleanAgent in background
        await trigger_data_cleaning(project_id)
        # Agent will emit progress events
        # Agent will set status to WAITING_CONFIRMATION when done
```

---

### Issue 3: Parse Might Not Be Actually Running
**Question**: "I don't know if it's actually parsing the text"

**How to Check**:
1. **Look at backend terminal logs** - You should see:
   ```
   ============================================================
   [PARSE INTENT] Starting for project: demo-project
   [PARSE INTENT] Prompt: Build a cat classifier...
   ============================================================
   [PromptParser] Using LangChain LLM...
   [PromptParser] Invoking LLM chain...
   [PromptParser] ✅ LLM response received in 3.2s
   ============================================================
   [PARSE INTENT] ✅ COMPLETED
   ============================================================
   ```

2. **Check frontend events** - Look for `PROMPT_PARSED` event with actual parsed data

3. **Watch for delay** - If parse is actually working, there should be a 2-5 second delay after clicking "Start Pipeline"

**If you DON'T see these logs**:
- The `/parse` endpoint might not be getting called
- Check browser Network tab for POST request to `/api/projects/demo-project/parse`
- Check for errors in browser console

---

## 🎯 **What Each Stage SHOULD Do**

### PARSE_INTENT ✅ (Working)
```
User clicks "Start Pipeline"
  → POST /api/projects/demo-project/parse
  → PromptParserAgent processes with LLM
  → Returns: {task_type, target, dataset_hint, constraints}
  → Sets stage to COMPLETED
  → Advances to DATA_SOURCE
```

### DATA_SOURCE ⏳ (Needs Work)
```
When DATA_SOURCE becomes IN_PROGRESS:
  → Show file upload UI OR
  → Use session.datasetLinks automatically
  → User uploads file OR confirms dataset
  → POST /api/projects/demo-project/upload
  → DataIngestionAgent processes file
  → Emits: DATASET_LOADED event
  → Sets stage to WAITING_CONFIRMATION
  → User confirms → Advances to DATA_CLEAN
```

### DATA_CLEAN ⏳ (Needs Work)
```
When DATA_CLEAN becomes IN_PROGRESS:
  → DataCleanAgent automatically runs
  → Analyzes dataset for:
    - Missing values
    - Outliers
    - Inconsistencies
  → Emits: DATA_PROFILE events (progress)
  → When done, emits: DATA_CLEANED event
  → Sets stage to WAITING_CONFIRMATION
  → User reviews cleaning results
  → User confirms → Advances to FEATURE_ENG
```

### FEATURE_ENG ⏳ (Needs Work)
```
When FEATURE_ENG becomes IN_PROGRESS:
  → FeatureEngineeringAgent runs
  → Suggests feature transformations
  → Applies: encoding, scaling, etc.
  → Emits: FEATURES_READY event
  → Sets stage to WAITING_CONFIRMATION
  → User confirms → Advances to MODEL_SELECT
```

### MODEL_SELECT ⏳ (Needs Work)
```
When MODEL_SELECT becomes IN_PROGRESS:
  → ModelSelectorAgent runs
  → Based on task_type, suggests models
  → Emits: MODEL_CANDIDATES event
  → Sets stage to WAITING_CONFIRMATION
  → User selects model
  → User confirms → Advances to TRAIN
```

### TRAIN ⏳ (Needs Work)
```
When TRAIN becomes IN_PROGRESS:
  → TrainingAgent starts training
  → Emits: TRAIN_PROGRESS events (epoch updates)
  → Shows: loss, accuracy, etc.
  → When done, emits: TRAIN_COMPLETED
  → Sets stage to WAITING_CONFIRMATION
  → User confirms → Advances to EVALUATE
```

### EVALUATE ⏳ (Needs Work)
```
When EVALUATE becomes IN_PROGRESS:
  → EvaluationAgent runs
  → Calculates metrics on test set
  → Emits: EVAL_METRICS event
  → Shows: accuracy, precision, recall, etc.
  → Sets stage to WAITING_CONFIRMATION
  → User confirms → Advances to EXPORT
```

### EXPORT ⏳ (Needs Work)
```
When EXPORT becomes IN_PROGRESS:
  → ExportAgent runs
  → Generates: model file, notebook, API spec
  → Emits: EXPORT_READY event
  → Sets stage to COMPLETED
  → Pipeline done!
```

---

## 🔧 **Immediate Fixes Needed**

### Priority 1: Fix Confirm Spam Issue
**File**: `backend/app/orchestrator/conductor.py`
**Add validation** to `confirm()` method

### Priority 2: Verify Parse is Actually Running
**Check**:
1. Backend terminal logs
2. Browser Network tab
3. Frontend console logs

### Priority 3: Wire Up Remaining Agents
**Each stage needs**:
1. Agent trigger when stage becomes IN_PROGRESS
2. Agent emits progress events
3. Agent sets WAITING_CONFIRMATION when done
4. Confirm button only enabled when WAITING_CONFIRMATION

---

## 📊 **Current vs. Desired State**

| Stage | Current Behavior | Desired Behavior |
|-------|-----------------|------------------|
| PARSE_INTENT | ✅ Calls LLM, parses prompt | ✅ Working correctly |
| DATA_SOURCE | ❌ Just waits for confirm | Should upload/load dataset |
| DATA_CLEAN | ❌ Just waits for confirm | Should analyze & clean data |
| FEATURE_ENG | ❌ Just waits for confirm | Should engineer features |
| MODEL_SELECT | ❌ Just waits for confirm | Should suggest models |
| TRAIN | ❌ Just waits for confirm | Should actually train model |
| EVALUATE | ❌ Just waits for confirm | Should evaluate metrics |
| EXPORT | ❌ Just waits for confirm | Should export artifacts |

---

## 🎬 **What You Should See (Ideal Flow)**

1. Click "Start Pipeline" → **2-5 sec delay** → PROMPT_PARSED event
2. Click "Confirm" → **DATA_SOURCE waits for upload**
3. Upload file → **Processing** → DATA_LOADED event
4. Click "Confirm" → **DATA_CLEAN runs** → Progress bars → CLEANED event
5. Click "Confirm" → **FEATURE_ENG runs** → Features engineered
6. Click "Confirm" → **MODEL_SELECT suggests** → Pick model
7. Click "Confirm" → **TRAIN shows progress** → Loss/accuracy updates
8. Click "Confirm" → **EVALUATE shows metrics** → F1, precision, recall
9. Click "Confirm" → **EXPORT generates** → Notebook, API ready

**vs. What You See Now:**
Click confirm 8 times instantly → All stages COMPLETED → No actual work done

---

## Want me to fix these issues?

I can:
1. ✅ Fix confirm validation (prevent spam)
2. ✅ Add proper stage-to-agent wiring
3. ✅ Make each stage do real work
4. ✅ Add proper WAITING_CONFIRMATION logic

Let me know!
