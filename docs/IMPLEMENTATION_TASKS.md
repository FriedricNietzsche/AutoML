# AutoML Agentic Builder - Implementation Tasks

This document outlines the step-by-step tasks required to build the "AutoML Agentic Builder" based on the product requirements. Use this guide to prompt an AI coding assistant or distribute work among team members.

## Phase 1: Foundation & Infrastructure

### Task 1.1: Backend Project Structure
**Goal:** Set up the FastAPI backend with the specific directory structure required.
- **Reference:** Repo structure section in requirements.
- **Action:** Create `backend/app` directories (api, ws, orchestrator, agents, ml, storage, events, db).
- **Files:** `backend/app/main.py`, `backend/requirements.txt`.
- **Details:** Initialize FastAPI app, include router placeholders, and set up CORS.

### Task 1.2: Frontend Project Structure
**Goal:** Set up the Next.js frontend with the specific directory structure required.
- **Reference:** Repo structure section in requirements.
- **Action:** Initialize Next.js 14 (App Router) with TypeScript, Tailwind, and ShadCN.
- **Files:** `frontend/src/app`, `frontend/src/components`, `frontend/src/lib`.
- **Details:** Install dependencies: `zustand`, `recharts`, `lucide-react`, `@monaco-editor/react`.

### Task 1.3: The Event Protocol & Types
**Goal:** Define the shared contract between frontend and backend.
- **Reference:** Section 3 & 5 (Contract & Events).
- **Backend:** Create `backend/app/events/schema.py` defining the Standard WebSocket Envelope and Event Enums (STAGE_STATUS, METRIC_SCALAR, etc.).
- **Frontend:** Create `frontend/src/lib/types.ts` mirroring these interfaces.
- **Critical:** Ensure `StageID` enum matches: PARSE_INTENT, DATA_SOURCE, PROFILE_DATA, PREPROCESS, MODEL_SELECT, TRAIN, REVIEW_EDIT, EXPORT.

### Task 1.4: WebSocket & Event Bus
**Goal:** Enable real-time communication.
- **Backend:** Implement `backend/app/events/bus.py` (in-memory pub/sub) and `backend/app/ws/hub.py` (WebSocket endpoint).
- **Frontend:** Implement `frontend/src/lib/ws.ts` to connect, handle reconnects, and dispatch events to the store.
- **Verification:** Users should be able to connect to `/ws/projects/{id}` and receive a "Hello" ping.

---

## Phase 2: Core Orchestration & UI Shell

### Task 2.1: The Conductor (State Machine)
**Goal:** Manage the 5-stage linked-list workflow.
- **Backend:** Implement `backend/app/orchestrator/conductor.py`.
- **Logic:**
  - Maintain current stage state (ID, Status).
  - Implement `transition_to(stage_id)`.
  - Implement `waiting_for_confirmation()` logic.
- **API:** detailed in `backend/app/api/stages.py` (GET state, POST confirm).

### Task 2.2: Workspace Layout UI
**Goal:** Build the 3-panel UI shell.
- **Frontend:** Create `/p/[projectId]/page.tsx`.
- **Components:**
  - `LeftPanel`: ChatPanel + StageTimeline.
  - `CenterPanel`: Dynamic view based on active stage.
  - `RightPanel`: FileExplorer + ArtifactsPanel.
- **State:** Use Zustand to store the current Project State Snapshot received from backend.

### Task 2.3: Chat & Timeline Components
**Goal:** Allow user interaction and status visibility.
- **Components:**
  - `ChatPanel.tsx`: Display user/system messages.
  - `StageTimeline.tsx`: Render the 5 stages as a vertical list. Highlight active stage. Show "Confirm" button if status is `WAITING_CONFIRMATION`.

---

## Phase 3: Stage 1 - Data & Intent

### Task 3.1: Prompt Parsing Agent
**Goal:** Interpret user intent.
- **Backend:** `backend/app/agents/prompt_parser.py`.
- **Logic:** Mock or use LLM to parse "Predict house prices" -> Task: REGRESSION, Target: "price".
- **Event:** Emit `PROMPT_PARSED`.

### Task 3.2: Data Ingestion & Storage
**Goal:** Handle CSV uploads and built-in datasets.
- **Backend:** `backend/app/storage/local_disk_store.py` and `backend/app/api/assets.py`.
- **Frontend:** `DatasetPreview.tsx` (Table view).
- **Action:** Support `POST /api/projects/{id}/upload`. Parse CSV with pandas, save to `data/projects/{id}/raw.csv`.
- **Event:** Emit `DATASET_SAMPLE_READY` with a JSON sample (head 5 rows).

### Task 3.3: Model Selection Agent
**Goal:** Choose candidate models based on task type.
- **Backend:** `backend/app/agents/model_selector.py`.
- **Logic:** If Regression -> Select LinearRegression, RandomForestRegressor.
- **Event:** Emit `MODEL_CANDIDATES`.

---

## Phase 4: Stage 2 - Profiling & Preprocessing

### Task 4.1: Data Profiling
**Goal:** Analyze the dataset.
- **Backend:** `backend/app/ml/tabular/profiling.py`.
- **Logic:** Calculate shape, missing values %, column types using Pandas.
- **Frontend:** `ProfilingPanel.tsx` (Charts for distribution, missingness).
- **Event:** Emit `PROFILE_SUMMARY` and generate assets for `TARGET_DISTRIBUTION_READY`.

### Task 4.2: Preprocessing Planner
**Goal:** Create a cleaning plan.
- **Backend:** `backend/app/agents/preprocess.py`.
- **Logic:**
  - Auto-detect: fill NA with mean/mode, OneHotEncode categoricals, StandardScaler numerics.
  - Use `sklearn.compose.ColumnTransformer`.
- **Event:** Emit `PREPROCESS_PLAN` (list of steps).

---

## Phase 5: Stage 3 - Training (The "Real" ML)

### Task 5.1: Training Loop & Runner
**Goal:** Train models and stream progress.
- **Backend:** `backend/app/ml/tabular/training.py`.
- **Critical:** Even for fast models, simulate progress updates so the UI looks cool (0% -> 100% over 2-3 seconds).
- **Logic:**
  - Split Train/Test.
  - Fit `Pipeline(preprocessor, model)`.
  - Calculate Metrics (RMSE/F1).
- **Thread Safety:** Run this in a background thread/task so it doesn't block the API.

### Task 5.2: Real-time Visualization
**Goal:** Show charts updating live.
- **Backend:** Emit `METRIC_SCALAR` events every "step".
- **Frontend:** `TrainingDashboard.tsx` using Recharts.
- **Chart:** Line chart listening to `metrics.loss` or `metrics.accuracy`.
- **Table:** Leaderboard showing different models comparing results.

### Task 5.3: Artifact Generation
**Goal:** Save models and plots.
- **Backend:** Save `.joblib` files. Generate Confusion Matrix / Residuals plots as PNGs.
- **Event:** Emit `ARTIFACT_ADDED` with storage URLs.

---

## Phase 6: Review & Export

### Task 6.1: Notebook Generation
**Goal:** Create a download code artifact.
- **Backend:** `backend/app/agents/reporter.py`.
- **Logic:** use `nbformat` (or templates) to generate a valid `.ipynb` that loads the CSV and runs the training code.
- **Event:** Emit `NOTEBOOK_READY`.

### Task 6.2: Final Export
**Goal:** Bundle everything.
- **Backend:** `backend/app/ml/tabular/export.py`.
- **Action:** Zip the model, preprocessor, and notebook.
- **Frontend:** `ArtifactsPanel.tsx` shows the download button.

---

## Phase 7: Polish & Demo Prep

### Task 7.1: Pre-canned Demo Mode
**Goal:** Ensure smooth presentation if live data fails.
- **Idea:** Create a "Demo Dataset" button that instantly loads the Titanic or Iris dataset and skips upload.

### Task 7.2: UI Refinement
**Goal:** Make it look high-tech/clean.
- **Frontend:** Apply ShadCN dark mode. Add easy-to-read colors for charts. Ensure the "Confirm" flow feels responsive.
