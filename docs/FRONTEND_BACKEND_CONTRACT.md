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