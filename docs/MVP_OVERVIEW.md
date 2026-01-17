# AutoML Agentic Backend — MVP Overview

## What’s Done
- **WebSocket & Events**: `/ws/projects/{id}` broadcasts HELLO + STAGE_STATUS and all contract events via an in-memory event bus.
- **Conductor (8 stages)**: PARSE_INTENT → DATA_SOURCE → PROFILE_DATA → PREPROCESS → MODEL_SELECT → TRAIN → REVIEW_EDIT → EXPORT with state endpoints (`GET /api/projects/{id}/state`, `POST /confirm`).
- **Intent & Model Selection**: `POST /api/projects/{id}/parse` (OpenRouter-backed prompt parser with fallback) emits `PROMPT_PARSED` and advances to DATA_SOURCE; `POST /api/projects/{id}/model/select` emits `MODEL_CANDIDATES`.
- **Assets**: `/api/assets/upload`, list, serve from `data/assets`.
- **Data Ingestion & Profiling**:
  - `POST /api/projects/{id}/upload` saves CSV, emits `DATASET_SAMPLE_READY`, then `PROFILE_SUMMARY`, `MISSINGNESS_TABLE_READY`, `TARGET_DISTRIBUTION_READY`, and a simple `PREPROCESS_PLAN`; conductor advances stages.
  - `POST /api/projects/{id}/ingest/kaggle` downloads a Kaggle dataset (needs `KAGGLE_USERNAME`/`KAGGLE_KEY`) and does the same.
- **Training**:
  - Tabular trainer (`/train/tabular`): sklearn pipeline with real fit/eval, streaming `TRAIN_PROGRESS`, `METRIC_SCALAR`, `TRAIN_RUN_FINISHED`.
  - Image trainer (`/train/image`): TF/Keras if available, otherwise synthetic streaming; streams training events and final metrics.
- **Agent set (LLM via LangChain + OpenRouter)**:
  - Prompt Parser (PARSE_INTENT) → `PROMPT_PARSED`
  - Model Selector (MODEL_SELECT) → `MODEL_CANDIDATES`
  - Reporter/Notebook Author (REVIEW_EDIT) → `NOTEBOOK_READY`, `REPORT_READY`
  - Optional Preprocess Planner (PREPROCESS) → richer `PREPROCESS_PLAN`
- **Deterministic flows**:
  - Data source: user upload; or Kaggle ingest using `dataset_hint`/prompt; curated defaults for demos.
  - Training decision: infer task from parser + schema (target detection, type inference); select heuristic model/pipeline if no confirmation needed.

## What’s Left for Demo/MVP
- Emit artifacts for training: confusion matrix/residuals/feature-importance files and `ARTIFACT_ADDED`/`CONFUSION_MATRIX_READY` events.
- Reporter/Notebook: generate notebook/report via LangChain/OpenRouter and emit `NOTEBOOK_READY`/`REPORT_READY`.
- Export bundle: zip model/preprocess/report/notebook and emit `EXPORT_READY`.
- Frontend wiring: ensure timeline/confirm and panels consume the above events (store is ready; needs real streams).

## Demo Scenarios
1) **Tabular Regression (Housing Prices)**
   - Parse intent: “predict house prices from features…”.
   - Ingest: upload CSV or Kaggle ingest (e.g., housing dataset).
   - Train: `/train/tabular` with target column; stream TRAIN_* events, metrics, residuals/feature-importance artifacts.
   - Review/Export: notebook/report + export zip.

2) **Time Series Forecasting (Weather/Electricity)**
   - Parse intent: “forecast electricity demand” (once TS runner is added).
   - Ingest: upload CSV or Kaggle ingest (public time-series dataset).
   - Train: time-series runner; stream metrics (RMSE/RMSLE) and forecast artifacts.
   - Review/Export: notebook/report + export zip.

3) **Vision Classification (Cats vs Dogs)**
   - Parse intent: “build a cat vs dog classifier”.
   - Ingest: public cats/dogs dataset or pre-bundled sample into `data/assets/projects/{id}/images`.
   - Train: `/train/image` (TF/PyTorch or synthetic fallback); stream TRAIN_* and confusion artifact.
   - Review/Export: notebook/report + export zip.

## Tech Stack & Usage
- **FastAPI** for HTTP/WS APIs.
- **Event Bus + WS** for real-time streaming (HELLO, STAGE_STATUS, TRAIN_*, artifacts).
- **Conductor** (in-memory) for stage state and transitions.
- **LangChain + OpenRouter** for intent parsing, model selection, reporter/notebook author (optional preprocess planner).
- **Pandas + sklearn** for tabular profiling/training; **TensorFlow/Keras** (optional) for vision training; synthetic stream fallback if TF absent.
- **Kaggle API** for dataset ingestion when provided creds.
- **Local assets** served from `data/assets` via `/api/assets`.

## Plan (Next Steps)
1) Data source automation: add Kaggle search fallback using `dataset_hint`; emit WAITING_CONFIRMATION when ambiguous.
2) Training artifacts: generate confusion/residual/feature-importance assets; emit `ARTIFACT_ADDED` + READY events.
3) Reporter/Notebook: LangChain/OpenRouter narrative + notebook generation; emit `NOTEBOOK_READY`/`REPORT_READY`.
4) Export: zip model/preprocess/notebook/report; emit `EXPORT_READY`.
5) Frontend verification: run end-to-end WS stream with real events for the three demo scenarios; ensure timeline/confirm flow updates.
