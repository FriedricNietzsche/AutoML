# AutoML Agentic Builder - Implementation Tasks (Team Split)

This document outlines step-by-step tasks to build the "AutoML Agentic Builder" for UofTHacks.  
We have **4 members**. Tasks are assigned so work can run in parallel with clean handoffs.

---

## Team Roles

### Member A — Frontend UI Lead (Primary UI/UX Owner)
**Goal:** Build the polished 3-panel workspace UI + stage visuals.  
**Owns:** UI layout, components, styling, charts, editor, responsiveness, demo polish.

### Member B — Frontend ↔ Backend Integrator (Dataflow + Contract Owner)
**Goal:** Make frontend consume backend state/events reliably.  
**Owns:** WebSocket client + reconnect, Zustand store, REST API client, shared types, staging/contract validation.

### Member C — Backend Orchestration + Dataflow (Backend Contract + Plumbing Owner)
**Goal:** Make backend expose stable APIs/events and manage stage state machine.  
**Owns:** FastAPI server, routers, stage machine, event bus, WS hub, storage, state snapshot endpoint.

### Member D — Backend AI/ML Agents + Modeling (ML + Agents Owner)
**Goal:** Implement real tabular ML pipeline and agent logic with artifact generation.  
**Owns:** prompt parser agent, dataset ingestion, profiling, preprocessing plan, training, notebook generation, export.

---

# Phase 0: Shared Agreement (30 min)
### ✅ Handoff Checkpoint 0.1 — Contract Freeze (A+B+C+D)
- Agree on:
  - Stage IDs + mapping to 5-stage linked-list
  - WebSocket envelope + event names
  - Asset URL rule (big things via HTTP, not WS)
- Output: `docs/FRONTEND_BACKEND_CONTRACT.md`

> **Owner:** Member B + Member C  
> **Reviewers:** Member A + Member D

---

# Phase 1: Foundation & Infrastructure

## Task 1.1: Backend Project Structure
**Owner:** Member C  
**Goal:** Set up FastAPI backend with required directory structure and router placeholders.
- Create `backend/app` directories: `api, ws, orchestrator, agents, ml, storage, events, db`.
- Files: `backend/app/main.py`, `backend/requirements.txt`.
- Initialize FastAPI app, CORS, include routers.

**Handoff:** Backend boots with `/health` endpoint.

---

## Task 1.2: Frontend Project Structure
**Owner:** Member A  
**Goal:** Set up Next.js 14 App Router with TypeScript, Tailwind, ShadCN.
- Create: `frontend/src/app`, `frontend/src/components`, `frontend/src/lib`.
- Install: `zustand`, `recharts`, `lucide-react`, `@monaco-editor/react`, `@supabase/supabase-js`.

**Handoff:** Frontend boots with a placeholder landing page.

---

## Task 1.3: Event Protocol & Shared Types
**Owner:** Member B (with Member C)  
**Goal:** Define shared contract types on both sides.
- Backend: `backend/app/events/schema.py` defines:
  - WS envelope
  - StageID enum (PARSE_INTENT, DATA_SOURCE, PROFILE_DATA, PREPROCESS, MODEL_SELECT, TRAIN, REVIEW_EDIT, EXPORT)
  - Event enums + payload typings
- Frontend: `frontend/src/lib/types.ts` mirrors exact interfaces.

**Handoff:** Frontend compiles with shared types and no TS errors.

---

## Task 1.4: WebSocket & Event Bus
**Owner:** Member C (backend) + Member B (frontend)
**Goal:** Enable real-time streaming from backend to frontend.
- Backend:
  - `backend/app/events/bus.py` in-memory pubsub
  - `backend/app/ws/hub.py` WebSocket `/ws/projects/{id}`
  - Emit a HELLO ping + STAGE_STATUS on connect
- Frontend:
  - `frontend/src/lib/ws.ts` connect + reconnect + dispatch to store

**Verification:** Frontend can connect and render “connected” event.

---

# Phase 2: Core Orchestration & UI Shell

## Task 2.1: Conductor (State Machine)
**Owner:** Member C  
**Goal:** Manage the 5-stage linked-list workflow + confirmation gates.
- Implement `backend/app/orchestrator/conductor.py`
- Maintain:
  - current stage id, status
  - plan pending vs approved
- Implement:
  - `transition_to(stage_id)`
  - `waiting_for_confirmation(stage_id)`
- API in `backend/app/api/stages.py`:
  - `GET /api/projects/{id}/state`
  - `POST /api/projects/{id}/confirm`

**Handoff:** Calling confirm advances stage and emits WS STAGE_STATUS + WAITING_CONFIRMATION events.

---

## Task 2.2: Workspace Layout UI (3-panel shell)
**Owner:** Member A  
**Goal:** Build the main UI layout and placeholders for stage panels.
- Create `/p/[projectId]/page.tsx`
- Layout:
  - Left: ChatPanel + StageTimeline
  - Center: StageViewContainer (dynamic)
  - Right: FileExplorer + ArtifactsPanel

**Handoff:** UI looks clean + navigable, even with dummy data.

---

## Task 2.3: Store + Timeline + Confirm Flow
**Owner:** Member B  
**Goal:** Connect UI shell to backend state/events via Zustand.
- Zustand store:
  - project state snapshot
  - event stream
  - stage statuses
- Components wired:
  - `StageTimeline` uses store
  - Confirm button calls backend `/confirm`
  - Chat sends messages to backend `/chat`

**Handoff:** Clicking confirm triggers backend and updates UI via WS.

---

# Phase 3: Stage 1 — Data & Intent

## Task 3.1: Prompt Parsing Agent
**Owner:** Member D  
**Goal:** Parse user prompt into structured intent.
- File: `backend/app/agents/prompt_parser.py`
- Mock parsing is OK initially; LLM optional.
- Emit event: `PROMPT_PARSED`.

**Handoff:** Prompt -> task_type, target guessed, constraints.

---

## Task 3.2: Data Ingestion & Dataset Sample Asset
**Owner:** Member C (storage/API) + Member D (pandas logic)  
**Goal:** Support CSV upload and built-in demo dataset selection.
- Backend:
  - `backend/app/storage/local_disk_store.py`
  - `backend/app/api/assets.py` (serve assets)
  - `POST /api/projects/{id}/upload` saves raw CSV
- Emit:
  - `DATASET_SAMPLE_READY` with `asset_url`, columns, n_rows

**Frontend**
- **Owner:** Member A
- Implement `DatasetPreview.tsx` to render sample table from asset_url.

**Handoff:** Upload -> dataset preview table works.

---

## Task 3.3: Model Selection Agent
**Owner:** Member D  
**Goal:** Choose candidate models by task.
- `backend/app/agents/model_selector.py`
- Emit: `MODEL_CANDIDATES`, `MODEL_SELECTED` on approval.

**Handoff:** Regression -> Linear/RF/GB; Classification -> Logistic/RF/GB.

---

# Phase 4: Stage 2 — Profiling & Preprocessing

## Task 4.1: Data Profiling (Artifacts + Events)
**Owner:** Member D  
**Goal:** Compute profiling summary and generate assets.
- `backend/app/ml/tabular/profiling.py`
- Compute:
  - shape, missing %, type breakdown, warnings
- Generate:
  - missingness table JSON asset
  - target distribution plot PNG asset
- Emit:
  - `PROFILE_PROGRESS`, `PROFILE_SUMMARY`
  - `MISSINGNESS_TABLE_READY`, `TARGET_DISTRIBUTION_READY`

**Frontend**
- **Owner:** Member A
- Build `ProfilingPanel.tsx`:
  - cards + table + plot display

**Handoff:** Profiling stage produces a “wow” panel.

---

## Task 4.2: Preprocessing Planner
**Owner:** Member D  
**Goal:** Create preprocessing pipeline plan and store config.
- `backend/app/agents/preprocess.py`
- ColumnTransformer plan:
  - numeric: impute mean + scale
  - categorical: impute most_frequent + onehot
- Emit: `PREPROCESS_PLAN`

**Handoff:** UI displays preprocessing steps clearly.

---

# Phase 5: Stage 3 — Training (The “WOW” Stage)

## Task 5.1: Training Runner (Real ML + Streaming)
**Owner:** Member D  
**Goal:** Train real sklearn pipelines and stream progress.
- `backend/app/ml/tabular/training.py`
- Split train/val
- Fit `Pipeline(preprocessor, model)`
- Real metrics computed:
  - classification: accuracy + f1 + confusion matrix
  - regression: rmse + r2 + residual plot
- Stream:
  - `TRAIN_RUN_STARTED`
  - repeated `TRAIN_PROGRESS`
  - repeated `METRIC_SCALAR` (simulate steps if model trains instantly)
  - `LEADERBOARD_UPDATED` if comparing models
  - `RESOURCE_STATS` (can be sampled or approximated)
  - `LOG_LINE`

**Backend non-blocking**
- Member C ensures this runs in background task/subprocess so API stays responsive.

**Handoff:** Training curves animate in real time.

---

## Task 5.2: Training Dashboard UI
**Owner:** Member A  
**Goal:** Real-time charts and leaderboard.
- `TrainingDashboard.tsx`:
  - line chart for loss
  - line chart for metric
  - leaderboard table
  - confusion matrix / residual plot area
  - resource cards
  - logs (optional here or in Console tab)

**Owner:** Member B  
- Wire METRIC_SCALAR streams into chart series in Zustand.

**Handoff:** UI feels like “training in progress” live.

---

## Task 5.3: Artifact Generation
**Owner:** Member D  
**Goal:** Save models + plots and notify frontend.
- Save:
  - model.joblib
  - report.json
  - confusion matrix PNG / residuals PNG
- Emit:
  - `ARTIFACT_ADDED` with URLs

**Owner:** Member A  
- `ArtifactsPanel.tsx` lists downloadable assets.

---

# Phase 6: Review & Export

## Task 6.1: Notebook Generation
**Owner:** Member D  
**Goal:** Generate notebook showing reproducible pipeline.
- `backend/app/agents/reporter.py`
- Use `nbformat` or a template notebook string.
- Emit: `NOTEBOOK_READY`, `REPORT_READY`.

**Frontend**
- **Owner:** Member A
- Show notebook download + preview (optional).

---

## Task 6.2: Export Bundle (ZIP)
**Owner:** Member D (export logic) + Member C (asset hosting)
**Goal:** Create final downloadable zip.
- `backend/app/ml/tabular/export.py`
- bundle:
  - model.joblib
  - pipeline/preprocessor.joblib
  - notebook.ipynb
  - report.json
- Emit: `EXPORT_READY`

**Frontend**
- **Owner:** Member A
- “Download ZIP” button, visible when ready.

---

# Phase 7: Polish & Demo Prep

## Task 7.1: Pre-canned Demo Mode
**Owner:** Member A + Member D  
**Goal:** Smooth demo even if uploads fail.
- Button: “Load Demo Dataset”
- loads Titanic/Iris (backend) and instantly shows sample/profile.

---

## Task 7.2: UI Refinement
**Owner:** Member A  
**Goal:** Make it feel like a premium AI product.
- dark mode
- nice transitions (loading states)
- stage confirmations feel responsive
- charts readable + smooth

---

# Integration Handoffs (Critical Checkpoints)

### ✅ Checkpoint 1 (Frontend can start early)
- Backend WS endpoint exists + sends HELLO + STAGE_STATUS
- Frontend WS connects and renders timeline updates

### ✅ Checkpoint 2 (Stage 1 demo)
- Upload dataset -> dataset preview renders

### ✅ Checkpoint 3 (Stage 2 demo)
- Profiling assets render

### ✅ Checkpoint 4 (Stage 3 WOW)
- Training events stream -> charts update live

### ✅ Checkpoint 5 (Final)
- Export zip download works

---
