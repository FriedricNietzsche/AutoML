# FRONTEND_BACKEND_CONTRACT.md

# Frontend and Backend Contract for AutoML Agentic Builder

## Stage IDs
- PARSE_INTENT
- DATA_SOURCE
- PROFILE_DATA
- PREPROCESS
- MODEL_SELECT
- TRAIN
- REVIEW_EDIT
- EXPORT

## Event Protocol
### Global Events
- **STAGE_STATUS**: {stage_id, status, message}
- **WAITING_CONFIRMATION**: {stage_id, summary, next_actions:[...]}
- **PLAN_PROPOSED**: {stage_id, plan_json}
- **PLAN_APPROVED**: {stage_id}
- **FILE_TREE_UPDATE**: {files:[{path,type,size,sha}]}
- **ARTIFACT_ADDED**: {artifact:{id,type,name,url,meta}}

### Stage-Specific Events
#### Stage 1: DATA COLLECTION / MODEL CHOICE
- **PROMPT_PARSED**: {task_type, target, dataset_hint, constraints}
- **DATASET_CANDIDATES**: {datasets:[{id,name,source,desc,meta}]}
- **DATASET_SELECTED**: {dataset_id}
- **MODEL_CANDIDATES**: {models:[{id,name,family,why,requirements}]}
- **MODEL_SELECTED**: {model_id}
- **DATASET_SAMPLE_READY**: {asset_url, columns, n_rows}

#### Stage 2: PROFILING / PREPROCESSING
- **PROFILE_PROGRESS**: {phase, pct}
- **PROFILE_SUMMARY**: {n_rows,n_cols,missing_pct,types_breakdown,warnings:[]}
- **MISSINGNESS_TABLE_READY**: {asset_url}
- **TARGET_DISTRIBUTION_READY**: {asset_url}
- **SPLIT_SUMMARY**: {train_rows,val_rows,test_rows,stratified,seed}
- **PREPROCESS_PLAN**: {steps:[...]}

#### Stage 3: TRAINING (RICH)
- **TRAIN_RUN_STARTED**: {run_id, model_id, metric_primary, config}
- **TRAIN_PROGRESS**: {run_id, epoch, epochs, step, steps, eta_s, phase}
- **METRIC_SCALAR**: {run_id, name, split, step, value}
- **LEADERBOARD_UPDATED**: {rows:[{model,params,metric,runtime_s}]}
- **BEST_MODEL_UPDATED**: {run_id, model_id, metric}
- **CONFUSION_MATRIX_READY**: {asset_url}
- **ROC_CURVE_READY**: {asset_url}
- **RESIDUALS_PLOT_READY**: {asset_url}
- **FEATURE_IMPORTANCE_READY**: {asset_url}
- **RESOURCE_STATS**: {run_id, cpu_pct, ram_mb, gpu_pct, vram_mb, step_per_sec}
- **LOG_LINE**: {run_id, level, text}
- **TRAIN_RUN_FINISHED**: {run_id, status, final_metrics}

## Stage 3 (TRAIN) — Gradient Descent / Stepwise Training Event Spec (v1)

This section standardizes **step-based** training updates so the UI can render:
- live loss curves (`METRIC_SCALAR`)
- progress bars (`TRAIN_PROGRESS`)
- diagnostics (optional) like weight norms and gradient norms

Even when training is performed by **non-iterative sklearn estimators**, the backend **MUST** emit these events by simulating steps, while computing **final metrics from real evaluation**.

### Goals / Guarantees
- Frontend can assume a monotonically increasing `(step)` sequence per `run_id`.
- A `TRAIN_RUN_STARTED` begins a run, and `TRAIN_RUN_FINISHED` terminates it.
- `METRIC_SCALAR` is the primary time-series signal for charts.
- For sklearn "one-shot" models, interim metrics may be **synthetic**, but `final_metrics` in `TRAIN_RUN_FINISHED` are **real**.

### Required ordering (per run_id)
1. `TRAIN_RUN_STARTED`
2. 1..N repetitions of:
   - `TRAIN_PROGRESS`
   - 0..K `METRIC_SCALAR`
   - optional diagnostics (`GD_DIAGNOSTIC`, `RESOURCE_STATS`, `LOG_LINE`)
3. Optional artifacts (e.g., `CONFUSION_MATRIX_READY`, `RESIDUALS_PLOT_READY`)
4. `TRAIN_RUN_FINISHED`

### Event: TRAIN_RUN_STARTED (required)
```json
{
  "run_id": "run_abc",
  "model_id": "random_forest",
  "metric_primary": "rmse",
  "config": {
    "task_type": "regression",
    "target": "price",
    "split": {"seed": 42, "test_size": 0.2},
    "preprocess": {"strategy": "auto"},
    "train": {"steps": 50, "epochs": 1}
  }
}
```

### Event: TRAIN_PROGRESS (required)
Represents the *timeline* of training. Use `steps` even if you do not have real epochs.
```json
{
  "run_id": "run_abc",
  "epoch": 1,
  "epochs": 1,
  "step": 17,
  "steps": 50,
  "eta_s": 3.2,
  "phase": "fit"
}
```
**Rules**
- `step` MUST start at 0 or 1 and increase by 1 until `steps`.
- `eta_s` may be null if unknown.
- `phase` enum: `"init"|"fit"|"eval"|"finalize"`.

### Event: METRIC_SCALAR (required)
Primary chart stream. Backend should emit at least:
- `loss` (synthetic allowed for non-iterative models)
- `metric_primary` (synthetic allowed mid-run; final must be real)
```json
{
  "run_id": "run_abc",
  "name": "loss",
  "split": "train",
  "step": 17,
  "value": 0.4321
}
```
**Fields**
- `name`: string (recommended: `"loss"`, `"rmse"`, `"r2"`, `"accuracy"`, `"f1"`)
- `split`: `"train"|"val"|"test"` (for mid-run typically `"train"`; final eval uses `"test"` or `"val"`)
- `step`: integer (align to `TRAIN_PROGRESS.step`)
- `value`: number

### Event: GD_DIAGNOSTIC (optional but standardized)
For true gradient descent models (or simulated diagnostics), emit extra scalars that the UI can show in an “Advanced” panel.
```json
{
  "run_id": "run_abc",
  "step": 17,
  "diagnostics": {
    "lr": 0.01,
    "weight_l2": 12.3,
    "grad_l2": 0.45,
    "update_l2": 0.012,
    "batch_size": 64
  }
}
```
**Notes**
- If using sklearn and not actually doing GD, you MAY omit this event entirely.

### Event: BEST_MODEL_UPDATED (recommended)
Emit when the current run becomes best by primary metric.
```json
{
  "run_id": "run_abc",
  "model_id": "random_forest",
  "metric": {"name": "rmse", "split": "val", "value": 12345.67}
}
```

### Event: TRAIN_RUN_FINISHED (required)
Final metrics MUST be computed from real model evaluation.
```json
{
  "run_id": "run_abc",
  "status": "success",
  "final_metrics": {
    "task_type": "regression",
    "primary": {"name": "rmse", "split": "test", "value": 12345.67},
    "metrics": [
      {"name": "rmse", "split": "test", "value": 12345.67},
      {"name": "r2", "split": "test", "value": 0.81}
    ]
  }
}
```
**status** enum: `"success"|"failed"|"cancelled"`.

---

## Stage 3 Simulation Rules (for non-iterative sklearn models)

To satisfy UI requirements for “live training”:
- Choose `steps` (default: **50**) and a wall-clock duration (default: **2–4 seconds**).
- Emit `TRAIN_PROGRESS` every step; sleep small intervals in a background worker.
- Emit `METRIC_SCALAR`:
  - `loss` as a smooth decreasing curve (e.g., exponential decay + noise).
  - `metric_primary` as a smooth improving curve (optional mid-run).
- After steps complete, perform real `.fit()` (if not already), then real evaluation, then emit:
  - final `METRIC_SCALAR` events for `metric_primary` on `"test"` at `step=steps`
  - `TRAIN_RUN_FINISHED` with real metrics

This preserves a consistent UX without lying about the final results.

---

#### Stage 4: REVIEW / EDIT
- **REPORT_READY**: {asset_url}
- **NOTEBOOK_READY**: {asset_url}
- **CODE_WORKSPACE_READY**: {files:[...]}
- **EDIT_SUGGESTIONS**: {suggestions:[...]}

#### Stage 5: EXPORT
- **EXPORT_PROGRESS**: {pct, message}
- **EXPORT_READY**: {asset_url, contents:[...], checksum}

## REST API Endpoints
### Project Management
- **GET /api/projects/{project_id}/state**: Returns the latest snapshot of the project state.

### User Confirmation
- **POST /api/projects/{project_id}/confirm**: Confirms the current stage and advances the pipeline.

### Asset Management
- **GET /api/projects/{project_id}/assets/{asset_id}**: Retrieves asset details by ID.

## Asset Rules
- Do not send large data over WebSocket. Instead, send asset URLs.
- Assets include:
  - Dataset sample JSON/CSV
  - Charts PNG
  - Missingness tables JSON
  - Confusion matrix PNG/JSON
  - Notebook.ipynb
  - Export.zip

This document serves as a contract between the frontend and backend teams to ensure alignment on the expected behavior and data structures throughout the development of the AutoML Agentic Builder project.