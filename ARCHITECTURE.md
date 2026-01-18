# AutoML AI Agent Architecture

## System Overview

AutoML is an **event-driven, AI-powered ML pipeline automation system** with real-time WebSocket communication.

```
┌────────────────────────────────────────────────────────────────┐
│                         USER                                    │
│                    (Browser @ localhost:5173)                   │
└─────────────────────┬──────────────────────────────────────────┘
                      │
                      ├─── HTTP Requests (Commands: start, confirm, upload)
                      │
                      └─── WebSocket Connection (/ws/projects/{id})
                            │
                            ↓ Real-time Events
┌────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + TypeScript)                │
├────────────────────────────────────────────────────────────────┤
│  • projectStore (Zustand)                                       │
│    - Subscribes to WebSocket events                            │
│    - Updates state: stages, datasets, metrics, artifacts       │
│  • Components                                                    │
│    - DashboardPane: Shows stage-specific visualizations        │
│    - DatasetImagePreview: 1-3 sample images                    │
│    - TrainingDashboard: Live metrics, loss curves              │
└─────────────────────┬──────────────────────────────────────────┘
                      │
                      │ WebSocket Events
                      ↓
┌────────────────────────────────────────────────────────────────┐
│                 BACKEND (FastAPI + Python)                      │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  API LAYER                                                │  │
│  │  • /api/demo/run/{project_id} - Trigger workflow         │  │
│  │  • /api/projects/{id}/parse - Parse prompt (optional)    │  │
│  │  • /api/projects/{id}/train - Train model                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  AI AGENTS (LangChain + HuggingFace Hub)                 │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  1. PromptParserAgent                              │  │  │
│  │  │     - Model: Llama 3.1 8B (via OpenRouter)        │  │  │
│  │  │     - Input: "Build cat/dog classifier"           │  │  │
│  │  │     - Output: {task_type: "vision",               │  │  │
│  │  │                 dataset_hint: "cat dog images"}   │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                           ↓                                │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  2. DatasetFinderAgent                             │  │  │
│  │  │     - API: HuggingFace Hub                         │  │  │
│  │  │     - Searches: 1000s of datasets                  │  │  │
│  │  │     - Filters: By task type + keywords             │  │  │
│  │  │     - License Validator:                           │  │  │
│  │  │       ✓ MIT, Apache, BSD, CC-BY allowed            │  │  │
│  │  │       ✗ GPL, proprietary rejected                  │  │  │
│  │  │     - Output: Top 3 datasets ranked by popularity  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                           ↓                                │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  3. ModelSelectorAgent                             │  │  │
│  │  │     - Task-aware model selection                   │  │  │
│  │  │     - For vision: CNN, ResNet, EfficientNet        │  │  │
│  │  │     - Provides pros/cons for each                  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  EVENT BUS (Pub/Sub Pattern)                             │  │
│  │  • In-memory event queue                                 │  │
│  │  • Subscribers: WebSocket connections by project_id      │  │
│  │  • Events: PROMPT_PARSED, DATASET_SELECTED,              │  │
│  │            TRAIN_PROGRESS, METRIC_SCALAR, etc.           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  CONDUCTOR (State Machine)                                │  │
│  │  • Manages 8-stage workflow:                             │  │
│  │    1. PARSE_INTENT → 2. DATA_SOURCE → 3. PROFILE_DATA    │  │
│  │    4. PREPROCESS → 5. MODEL_SELECT → 6. TRAIN            │  │
│  │    7. REVIEW_EDIT → 8. EXPORT                            │  │
│  │  • Emits STAGE_STATUS events                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  WEBSOCKET HUB (/ws/projects/{project_id})                │  │
│  │  • ConnectionManager                                      │  │
│  │  • Broadcasts events to connected clients                │  │
│  │  • Auto-reconnect support                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                          ↑                       │
│                                          │                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ML EXECUTION (Trainers)                                  │  │
│  │  • TabularTrainer: sklearn pipelines                     │  │
│  │  • ImageTrainer: TF/PyTorch CNNs                          │  │
│  │  • Emits: TRAIN_PROGRESS, METRIC_SCALAR, GD_PATH_UPDATE  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Event Flow Example: "Build cat/dog classifier"

```
1. USER types prompt → Frontend sends POST /api/demo/run/demo
2. PromptParserAgent (LangChain):
   Input: "Build cat/dog classifier"
   Output: {task_type: "vision", dataset_hint: "cat dog images"}
   Event: PROMPT_PARSED → WebSocket → Frontend

3. DatasetFinderAgent (HF API):
   Searches: task="vision" keywords="cat dog images"
   Results: [microsoft/cats_vs_dogs, Oxford-IIIT Pet, ...]
   License Check: microsoft/cats_vs_dogs ✓ (MIT)
   Event: DATASET_CANDIDATES → WebSocket → Frontend
   Event: DATASET_SELECTED → WebSocket → Frontend

4. Data Ingestion:
   Downloads: 30 images from microsoft/cats_vs_dogs
   Event: DATASET_SAMPLE_READY {images: [url1, url2, url3]} → WebSocket
   Frontend: DashboardPane shows 3 images in grid

5. ModelSelectorAgent:
   Recommends: CNN (best for small datasets)
   Event: MODEL_CANDIDATES → WebSocket → Frontend

6. Training:
   TabularTrainer starts
   Every epoch:
     Event: TRAIN_PROGRESS {step: X, steps: 50} → WebSocket
     Event: METRIC_SCALAR {name: "loss", value: 0.42} → WebSocket
     Event: GD_PATH_UPDATE {points: [...]} → WebSocket
   Frontend: Dashboard live-updates loss curve

7. Completion:
   Event: TRAIN_RUN_FINISHED → WebSocket
   Event: NOTEBOOK_READY → WebSocket
   Event: EXPORT_READY → WebSocket
   Frontend: Shows download button
```

## Model Capabilities

### LangChain Agent (Prompt Parser)
- **Model**: Llama 3.1 8B Instruct (via OpenRouter)
- **Provider**: OpenRouter.ai (free tier)
- **Capabilities**:
  - Intent classification (5 task types)
  - Entity extraction (dataset hints, constraints)
  - Structured JSON output (Pydantic schemas)
  - Fallback to heuristics if API fails

### HuggingFace Hub Integration
- **API**: HuggingFace Hub Python SDK
- **Access**: 1000 requests/hour (free tier)
- **Features**:
  - Full-text dataset search
  - Metadata extraction (license, downloads, size)
  - Popularity-based ranking
  - License compliance checking

### Dataset License Validator
**Allowed Licenses:**
- MIT, Apache-2.0
- BSD-2-Clause, BSD-3-Clause
- CC-BY-4.0, CC-BY-SA-4.0, CC0-1.0
- CC-BY-NC-4.0 (research/demo only)

**Rejected Licenses:**
- GPL, AGPL (copyleft)
- Proprietary, Commercial-only
- CC-BY-ND (no derivatives)
- Unknown/unspecified

## Frontend-Backend Interaction

### Communication Channels

**1. HTTP (Commands)**
```typescript
// Frontend sends commands
POST /api/demo/run/{project_id}?prompt=...
POST /api/projects/{id}/confirm
POST /api/projects/{id}/upload
```

**2. WebSocket (State Updates)**
```typescript
// Frontend subscribes
ws://localhost:8000/ws/projects/{project_id}

// Receives events
{
  "type": "event",
  "event": {
    "name": "DATASET_SAMPLE_READY",
    "payload": {
      "images": ["url1", "url2", "url3"],
      "columns": [...],
      "n_rows": 1000
    }
  }
}
```

### State Management Flow

```typescript
// projectStore.ts (Zustand)
const projectStore = create((set) => ({
  // State
  stages: {},
  datasetSample: null,
  trainingMetrics: null,
  
  // WebSocket handler
  applyEvent: (evt) => {
    if (evt.event?.name === 'DATASET_SAMPLE_READY') {
      set({ datasetSample: evt.event.payload });
    }
    if (evt.event?.name === 'METRIC_SCALAR') {
      set((state) => ({
        trainingMetrics: {
          ...state.trainingMetrics,
          metricsHistory: [...history, newMetric]
        }
      }));
    }
  }
}));

// Component
function DashboardPane() {
  const { datasetSample, currentStageId } = useProjectStore();
  
  if (currentStageId === 'DATA_SOURCE' && datasetSample?.images) {
    return <DatasetImagePreview images={datasetSample.images} />;
  }
}
```

## Key Design Decisions

1. **WebSocket as Single Source of Truth**
   - All state changes flow through WS
   - HTTP only for commands, not queries
   - Ensures frontend never out of sync

2. **Event-Driven Architecture**
   - Loose coupling between components
   - Easy to add new features (just emit new events)
   - Testable (can inject mock events)

3. **AI Agent Fallbacks**
   - LangChain fails → heuristic parser
   - HF API down → cached popular datasets
   - No valid license → prompt user for upload

4. **License-First Approach**
   - Never use data without explicit permission
   - Conservative rejection of unknown licenses
   - Audit trail for compliance

## Performance & Scaling

**Current Limits:**
- OpenRouter: ~100 requests/day (free tier)
- HuggingFace: 1000 requests/hour
- WebSocket: 100 concurrent connections
- Training: In-memory (single machine)

**Future Improvements:**
- Redis for event bus (multi-instance)
- Celery for async training queues
- S3 for dataset caching
- Ollama for local LLM (remove OpenRouter dep)

---

**Summary**: AutoML uses AI agents to automate the entire ML pipeline, from understanding natural language prompts to finding compliant datasets and streaming live training updates to the user's browser via WebSocket.
