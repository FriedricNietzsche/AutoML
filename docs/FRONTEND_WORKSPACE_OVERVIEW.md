# AutoML Frontend & Workspace Deep Dive

This document explains how the frontend is structured, how the workspace runs, what the visualizers expect, and how the mock streams feed them. It is written so someone new to the repo can follow the data flow end‑to‑end.

## High-level flow
- **Entry point:** `HomePage` collects a goal prompt (and optional Kaggle link) then creates a `BuildSession` via `createBuildSession` (stored in localStorage under `autoai.buildSession.current`). The router then navigates to `/workspace`.
- **Workspace guard:** `WorkspacePage` loads the persisted session (`getCurrentSession`). If none exists, it redirects back to `/`.
- **App shell:** `AppShell` is the main workspace frame. It wires theme toggling, connects to the project WebSocket, manages the virtual file system (VFS), panel sizes, tabs, and runs the mock pipeline when the user clicks “Run”.

## AppShell layout & state (`frontend/src/components/shell/AppShell.tsx`)
- **Panels:** Three resizable columns:
  - Left: `AIBuilderPanel` (session summary + chat/change requests).
  - Center: tabbed workspace (`WorkspaceTabs`) showing Preview, Dashboard, Publishing, Console, or file editors.
  - Right: `FilesPanel` rendering the VFS tree.
- **TopBar:** Run/Generate/Export buttons, theme toggle, and WebSocket status/ping.
- **VFS:** Initialized from `initialFileSystem` (see `frontend/src/lib/mockData.ts`). Files live in localStorage (`vfs.files`). `updateFileContent`/`getFileContent` walk the tree and persist updates (artifacts, logs, configs).
- **WebSocket connection:** `useProjectStore` connects to a backend WS (or mock) via `connectProject`. Incoming events are also appended to `/logs/training.log` for visibility.
- **Pipeline runner:** `usePipelineRunner` just marks `isRunning` and clears logs; the visualizer drives most of the simulated work. When Preview finishes, `completePipeline` clears the running flag and the session status is set to `ready`.
- **Tabs:** Default tabs are Preview, Dashboard, Publishing, Console; file tabs are added when a file is clicked in the Files panel.
- **Session guard:** If the session disappears, AppShell navigates back to `/`.

## Preview tab & visualizer
- **PreviewPane:** Switches between:
  - `EmptyPreview` when no session,
  - `TrainingLoaderV2` (alias of `TrainingLoader`) while building/running,
  - `APIDocsPane` after the loader signals completion.
- It receives `files`, `isRunning`, `updateFileContent`, and `onComplete` to mark the simulation done and flip to docs view.

## TrainingLoader (core visualizer) (`frontend/src/components/center/loader/TrainingLoader.tsx`)
- **Stage state:** `currentStage` (0 = idle, 1‑5 = stages), `isStageRunning`, `stageCompleted`. “Proceed” starts stage 1 and advances when a stage finishes.
- **Stage definitions:** Five animated stages (the “fixed nodes” at the bottom mirror these):
  1) Data Load, 2) Preprocess, 3) Train, 4) Evaluate, 5) Export.
  - Each stage pulls one `StepDef` from `stageDefinitions` with `phases` of type `operation`, `graph`, or `visual`. These drive which visual block renders.
- **Data inputs (from `useMockAutoMLStream`):**
  - `metricsState.lossSeries`, `accSeries`, `f1Series`, `rmseSeries`: used in graph phases.
  - `metricsState.datasetPreview`: either tabular rows/columns or image data. Image previews trigger the “vectorization” animation; tabular previews feed the grid animation.
  - `metricsState.embeddingPoints`, `gradientPath`, `residuals`, `confusionTable`: feed the specialized visuals.
  - `metricsState.metricsSummary`: supplies accuracy/f1/rmse summary cards.
  - `thinkingByStage[stage]`: text log bubbles in Stage 1.
  - `leaderboard`: drives “best model” data in training visuals.
- **Artifacts/logs:** As stages start, the loader writes artifacts into the VFS (e.g., `/artifacts/progress.json`) and appends user/AI logs to `/logs/training.log`. When training ends it writes updated artifacts handed in via `updateFileContent`.
- **Fixed linked-list nodes:** The bottom row renders five circular nodes with connectors. Colors: accent when active, success when done, border-muted when pending. Arrows are simple div triangles; connectors change color with progress.
- **Animations:** Driven by an RAF clock (`nowRef`) and `phaseProgress`. Reduced-motion preferences disable complex motion/spinners.

## Dashboard, Console, Publishing, Files
- **DashboardPane:** Reads JSON artifacts directly from the VFS:
  - `/config/pipeline.json` (pipeline nodes with `{id,label,status,progress}`) renders the pipeline chips/connectors.
  - `/artifacts/metrics.json` and `/artifacts/confusion_matrix.json` back the mini-metrics and confusion matrix cards.
- **ConsolePane:** Shows `/logs/training.log`.
- **PublishingPane:** Static placeholder content for publishing/export concepts.
- **FilesPanel:** Renders the VFS tree (`files` from AppShell) and opens file tabs on click.

## Virtual File System seed (`frontend/src/lib/mockData.ts`)
- **Pipeline config:** `/config/pipeline.json` starts with 8 pending nodes:
  1) data_import, 2) validation, 3) eda, 4) preprocessing, 5) feature_eng, 6) training, 7) evaluation, 8) export.
  AIBuilderDashboard expects this shape: `{ id, label, status, progress, logs }`.
- **Other seeds:** dataset.json, model.json, artifacts/metrics.json, artifacts/confusion_matrix.json, training_log.txt, README, etc. These mirror what the loader writes to during a run.

## Mock stream consumer (`frontend/src/mock/useMockAutoMLStream.ts`)
- Wraps `createMockAutoMLStream` (generator) and batches incoming `MockWSEnvelope` events via `requestAnimationFrame`.
- **MetricsState shape:** 
  - Scalars/series: `lossSeries`, `accSeries`, `f1Series`, `rmseSeries`.
  - Tables/points: `confusionTable`, `embeddingPoints`, `gradientPath`, `residuals`.
  - Metadata: `surfaceSpec`, `pipelineGraph` (nodes/edges), `leaderboard`, `metricsSummary`.
  - UX: `thinkingByStage` (derived from `LOG_LINE` messages prefixed with `THINKING:`).
  - Data preview: `datasetPreview` holds either tabular rows/columns or imageData placeholder.
- **Event handling (`reduceMetricsState`):**
  - `ARTIFACT_ADDED` with meta.kind drives special setters:
    - `loss_surface` → `surfaceSpec`
    - `gradient_path` → `gradientPath`
    - `embedding_points` → `embeddingPoints`
    - `residuals` → `residuals`
    - `pipeline_graph` → `pipelineGraph` (expects `{ nodes, edges }`)
  - `LOG_LINE` starting with `THINKING:` → stored under the current `stage.id`.
  - `METRIC_SCALAR` → builds/updates series (loss/acc/f1/rmse).
  - `DATASET_PREVIEW` → tabular or image preview (image uses client-side load).
  - `PROFILE_SUMMARY` / `METRIC_TABLE` (missingness) → stored; confusion matrix is picked up from `confusion` or `METRIC_TABLE`.
  - `LEADERBOARD_UPDATED` → `leaderboard` entries.
  - Everything batches through `pendingEventsRef` to avoid render thrash.

## Mock event producer (`frontend/src/mock/mockBackendStream.ts`)
- **Generators:**
  - `createLegacyMockEventStream` yields **legacy** `BackendEvent` objects for steps S0–S7.
  - `createMockAutoMLStream` wraps legacy events into **contracted** `MockWSEnvelope` objects `{ type: 'EVENT', stage: {id,index,status}, event: {name,payload} }`, mapping step → stage via `STEP_TO_STAGE` (PARSE_INTENT → EXPORT).
- **Stage/status mapping:** `stageStatusFromStep` converts legacy step statuses into contract `StageStatus` (`IN_PROGRESS`, `WAITING_CONFIRMATION`, `COMPLETED`). `STAGE_ORDER` comes from `frontend/src/lib/contract.ts`.
- **Timeline (legacy stream):**
  - S0 Parse Intent: PLAN_PROPOSED/PLAN_SELECTED, THINKING logs, task artifact.
  - S1 Data Source: DATASET_SEARCH_RESULTS, DATASET_INGESTED, THINKING logs.
  - S2 Profile Data: PROFILE_PROGRESS, DATASET_PREVIEW (tabular unless scenario B → image placeholder), PROFILE_SUMMARY, METRIC_TABLE (missingness).
  - S3 Preprocess/Feature Plan: PLAN_PROPOSED/PLAN_SELECTED, PIPELINE_GRAPH from scenario (nodes/edges), FEATURE_SUMMARY, config artifacts.
  - S4 Model Select: PLAN_PROPOSED/PLAN_SELECTED, LEADERBOARD_UPDATED, model plan artifact.
  - S5 Train: TRAIN_PROGRESS per epoch, METRIC_SCALAR (train/val loss, accuracy/f1 or rmse), METRIC_TABLE (confusion), LEADERBOARD_UPDATED, artifacts (metrics, model, embedding_points, residuals if present).
  - S6 Report: REPORT_READY, report/model card artifacts.
  - S7 Export: EXPORT_READY and serving spec artifact.
- **Scenarios (`frontend/src/mock/scenarios/*.ts`):** Provide deterministic data for curves, embeddings, residuals, and pipeline graphs. Example pipeline graphs:
  - Scenario A: ingest → clean → encode → train → eval (5 nodes).
  - Scenario B: ingest → profile → encode → train → calibrate → eval (6 nodes).
  - Scenario C: ingest → scale → train → eval (4 nodes).
- **Contract envelope mode (second half of file):** Also defines a richer `runTimeline` that emits contract-style events (PLAN_PROPOSED, PROFILE_PROGRESS, PIPELINE_GRAPH, RESOURCE_STATS, etc.) with payloads matching `frontend/src/lib/contract.ts` types. This uses async `waitForConfirm`/`waitForPlan` guards and a queue-based stream.

## Data expectations for visualizers
- **Training graphs:** Expect `METRIC_SCALAR` values keyed by `metric` (`train_loss`, `val_loss`, `accuracy`, `f1`, `rmse`) and epochs. Loss curves require both train/val entries for the same epoch.
- **Data preview:** 
  - Tabular: `rows: number[][]`, `columns: string[]`.
  - Image: `imageData` placeholder `{ width, height, pixels: [], needsClientLoad: true }`; the component samples the image client-side.
- **Embedding scatter:** Uses `embeddingPoints` artifacts (`[{id,x,y,label,weight}]`).
- **Confusion matrix:** Uses `METRIC_TABLE` with `table === 'confusion'` or residual artifacts producing `confusion_table`.
- **Pipeline graph:** Requires `{ nodes: {id,label}[...], edges: {from,to}[...] }`. Currently consumed in Dashboard; TrainingLoader does not render it directly.
- **Fixed stage nodes:** Derived from local `fixedNodes` (currently 5) rather than streamed data; they mirror the stageDefinitions list.

## Workspace data loop in practice
1) User starts run → `usePipelineRunner` sets `isRunning` → Preview switches to `TrainingLoader`.
2) `TrainingLoader` subscribes to `useMockAutoMLStream` (enabled when `currentStage > 0`).
3) Streamed envelopes update `metricsState` and append logs/artifacts via `applyProjectEvent` and `updateFileContent`.
4) Visuals react to `metricsState` (series, tables, previews, embeddings).
5) When the last stage finishes, `onComplete` tells Preview to show `APIDocsPane` and `completePipeline` clears the running flag. Session status becomes `ready`.

## Where to plug in real data
- Replace `createMockAutoMLStream` with a real WS stream that emits envelopes matching `WSEnvelope` in `frontend/src/lib/contract.ts`.
- Ensure payloads populate:
  - `METRIC_SCALAR` (train/val loss, accuracy/f1/rmse),
  - `METRIC_TABLE` (confusion/missingness),
  - `DATASET_PREVIEW` (tabular or image),
  - `ARTIFACT_ADDED` meta with `kind` values the reducer understands,
  - `PIPELINE_GRAPH` nodes/edges if you want Dashboard to reflect live pipelines,
  - `LOG_LINE` messages prefixed with `THINKING:` for Stage 1 text stream.
- VFS updates should still go through `updateFileContent` (artifact paths are stable: `/artifacts/*.json`, `/config/*.json`, `/logs/training.log`).

## Reference map
- Workspace page guard: `frontend/src/pages/WorkspacePage.tsx`
- App shell: `frontend/src/components/shell/AppShell.tsx`
- Preview: `frontend/src/components/center/PreviewPane.tsx`
- Visualizer: `frontend/src/components/center/loader/TrainingLoader.tsx`
- Mock stream consumer: `frontend/src/mock/useMockAutoMLStream.ts`
- Mock stream producer: `frontend/src/mock/mockBackendStream.ts`
- Mock scenarios: `frontend/src/mock/scenarios/*.ts`
- Event/contract types: `frontend/src/lib/contract.ts`, `frontend/src/mock/backendEventTypes.ts`
- VFS seed: `frontend/src/lib/mockData.ts`
