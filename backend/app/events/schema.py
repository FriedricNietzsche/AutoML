from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from enum import Enum

# ============================================================================
# STAGE DEFINITIONS
# ============================================================================

class StageID(str, Enum):
    """Stage identifiers for the 5-stage workflow"""
    PARSE_INTENT = "PARSE_INTENT"
    DATA_SOURCE = "DATA_SOURCE"
    PROFILE_DATA = "PROFILE_DATA"
    PREPROCESS = "PREPROCESS"
    MODEL_SELECT = "MODEL_SELECT"
    TRAIN = "TRAIN"
    REVIEW_EDIT = "REVIEW_EDIT"
    EXPORT = "EXPORT"


class StageStatus(str, Enum):
    """Status of each stage"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# Stage order helper so we can attach the canonical index to envelopes.
STAGE_SEQUENCE: List[StageID] = [
    StageID.PARSE_INTENT,
    StageID.DATA_SOURCE,
    StageID.PROFILE_DATA,
    StageID.PREPROCESS,
    StageID.MODEL_SELECT,
    StageID.TRAIN,
    StageID.REVIEW_EDIT,
    StageID.EXPORT,
]

# Map 8 backend stages into 5 frontend visualization buckets.
VIS_STAGE_INDEX = {
    StageID.PARSE_INTENT: 0,
    StageID.DATA_SOURCE: 0,
    StageID.PROFILE_DATA: 1,
    StageID.PREPROCESS: 1,
    StageID.MODEL_SELECT: 2,  # dedicated model-select node
    StageID.TRAIN: 3,
    StageID.REVIEW_EDIT: 4,
    StageID.EXPORT: 4,
}


def stage_index(stage_id: StageID) -> int:
    """Return the stable index for a given stage id (mapped to 5-stage visual timeline)."""
    return VIS_STAGE_INDEX.get(stage_id, 0)


# ============================================================================
# EVENT TYPE DEFINITIONS
# ============================================================================

class EventType(str, Enum):
    """All possible event types in the system"""

    # Connection lifecycle
    HELLO = "HELLO"
    
    # Global Events
    STAGE_STATUS = "STAGE_STATUS"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    PLAN_PROPOSED = "PLAN_PROPOSED"
    PLAN_APPROVED = "PLAN_APPROVED"
    FILE_TREE_UPDATE = "FILE_TREE_UPDATE"
    ARTIFACT_ADDED = "ARTIFACT_ADDED"
    
    # Stage 1: DATA COLLECTION / MODEL CHOICE
    PROMPT_PARSED = "PROMPT_PARSED"
    DATASET_CANDIDATES = "DATASET_CANDIDATES"
    DATASET_SELECTED = "DATASET_SELECTED"
    DATASET_SEARCH_FAILED = "DATASET_SEARCH_FAILED"
    DATASET_LOAD_FAILED = "DATASET_LOAD_FAILED"
    MODEL_CANDIDATES = "MODEL_CANDIDATES"
    MODEL_SELECTED = "MODEL_SELECTED"
    DATASET_SAMPLE_READY = "DATASET_SAMPLE_READY"
    
    # Stage 2: PROFILING / PREPROCESSING
    PROFILE_PROGRESS = "PROFILE_PROGRESS"
    PROFILE_SUMMARY = "PROFILE_SUMMARY"
    MISSINGNESS_TABLE_READY = "MISSINGNESS_TABLE_READY"
    TARGET_DISTRIBUTION_READY = "TARGET_DISTRIBUTION_READY"
    SPLIT_SUMMARY = "SPLIT_SUMMARY"
    PREPROCESS_PLAN = "PREPROCESS_PLAN"
    
    # Stage 3: TRAINING (RICH)
    TRAIN_RUN_STARTED = "TRAIN_RUN_STARTED"
    TRAIN_PROGRESS = "TRAIN_PROGRESS"
    METRIC_SCALAR = "METRIC_SCALAR"
    LEADERBOARD_UPDATED = "LEADERBOARD_UPDATED"
    BEST_MODEL_UPDATED = "BEST_MODEL_UPDATED"
    CONFUSION_MATRIX_READY = "CONFUSION_MATRIX_READY"
    ROC_CURVE_READY = "ROC_CURVE_READY"
    RESIDUALS_PLOT_READY = "RESIDUALS_PLOT_READY"
    FEATURE_IMPORTANCE_READY = "FEATURE_IMPORTANCE_READY"
    RESOURCE_STATS = "RESOURCE_STATS"
    LOG_LINE = "LOG_LINE"
    TRAIN_RUN_FINISHED = "TRAIN_RUN_FINISHED"
    
    # Stage 4: REVIEW / EDIT
    REPORT_READY = "REPORT_READY"
    NOTEBOOK_READY = "NOTEBOOK_READY"
    CODE_WORKSPACE_READY = "CODE_WORKSPACE_READY"
    EDIT_SUGGESTIONS = "EDIT_SUGGESTIONS"
    
    # Stage 5: EXPORT
    EXPORT_PROGRESS = "EXPORT_PROGRESS"
    EXPORT_READY = "EXPORT_READY"


# ============================================================================
# WEBSOCKET ENVELOPE & CORE MODELS
# ============================================================================

class Stage(BaseModel):
    """Stage information"""
    id: StageID
    index: int
    status: StageStatus


class EventPayload(BaseModel):
    """Generic event payload wrapper"""
    name: EventType
    payload: Dict[str, Any]


class EventMessage(BaseModel):
    """WebSocket message envelope - wraps all events"""
    v: int = 1  # Protocol version
    type: str = "EVENT"  # Message type (usually "EVENT")
    project_id: str
    seq: int  # Sequence number (monotonic per project)
    ts: int  # Timestamp (unix milliseconds)
    stage: Stage
    event: EventPayload


class WSEnvelope(EventMessage):
    """
    Compatibility alias used throughout the codebase.
    """
    pass


# ============================================================================
# EVENT PAYLOAD SCHEMAS (Typed Payloads)
# ============================================================================

# --- Global Event Payloads ---

class StageStatusPayload(BaseModel):
    stage_id: StageID
    status: StageStatus
    message: str


class WaitingConfirmationPayload(BaseModel):
    stage_id: StageID
    summary: str
    next_actions: List[str]


class PlanProposedPayload(BaseModel):
    stage_id: StageID
    plan_json: Dict[str, Any]


class PlanApprovedPayload(BaseModel):
    stage_id: StageID


class FileInfo(BaseModel):
    path: str
    type: str
    size: int
    sha: Optional[str] = None


class FileTreeUpdatePayload(BaseModel):
    files: List[FileInfo]


class ArtifactInfo(BaseModel):
    id: str
    type: str
    name: str
    url: str
    meta: Optional[Dict[str, Any]] = None


class ArtifactAddedPayload(BaseModel):
    artifact: ArtifactInfo


# --- Stage 1: DATA COLLECTION / MODEL CHOICE ---

class PromptParsedPayload(BaseModel):
    task_type: str
    target: Optional[str] = None
    dataset_hint: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None


class DatasetInfo(BaseModel):
    id: str
    name: str
    source: str
    desc: str
    meta: Optional[Dict[str, Any]] = None


class DatasetCandidatesPayload(BaseModel):
    datasets: List[DatasetInfo]


class DatasetSelectedPayload(BaseModel):
    dataset_id: str


class ModelInfo(BaseModel):
    id: str
    name: str
    family: str
    why: str
    requirements: Optional[Dict[str, Any]] = None


class ModelCandidatesPayload(BaseModel):
    models: List[ModelInfo]


class ModelSelectedPayload(BaseModel):
    model_id: str


class DatasetSampleReadyPayload(BaseModel):
    asset_url: str
    columns: List[str]
    n_rows: int


# --- Stage 2: PROFILING / PREPROCESSING ---

class ProfileProgressPayload(BaseModel):
    phase: str
    pct: float


class ProfileSummaryPayload(BaseModel):
    n_rows: int
    n_cols: int
    missing_pct: float
    types_breakdown: Dict[str, int]
    warnings: List[str]


class MissingnessTableReadyPayload(BaseModel):
    asset_url: str


class TargetDistributionReadyPayload(BaseModel):
    asset_url: str


class SplitSummaryPayload(BaseModel):
    train_rows: int
    val_rows: int
    test_rows: int
    stratified: bool
    seed: int


class PreprocessPlanPayload(BaseModel):
    steps: List[Dict[str, Any]]


# --- Stage 3: TRAINING (RICH) ---

class TrainRunStartedPayload(BaseModel):
    run_id: str
    model_id: str
    metric_primary: str
    config: Dict[str, Any]


class TrainProgressPayload(BaseModel):
    run_id: str
    epoch: int
    epochs: int
    step: int
    steps: int
    eta_s: float
    phase: str


class MetricScalarPayload(BaseModel):
    run_id: str
    name: str
    split: str
    step: int
    value: float


class LeaderboardRow(BaseModel):
    model: str
    params: Dict[str, Any]
    metric: float
    runtime_s: float


class LeaderboardUpdatedPayload(BaseModel):
    rows: List[LeaderboardRow]


class BestModelUpdatedPayload(BaseModel):
    run_id: str
    model_id: str
    metric: float


class ConfusionMatrixReadyPayload(BaseModel):
    asset_url: str


class ROCCurveReadyPayload(BaseModel):
    asset_url: str


class ResidualsPlotReadyPayload(BaseModel):
    asset_url: str


class FeatureImportanceReadyPayload(BaseModel):
    asset_url: str


class ResourceStatsPayload(BaseModel):
    run_id: str
    cpu_pct: float
    ram_mb: float
    gpu_pct: Optional[float] = None
    vram_mb: Optional[float] = None
    step_per_sec: Optional[float] = None


class LogLinePayload(BaseModel):
    run_id: str
    level: str
    text: str


class TrainRunFinishedPayload(BaseModel):
    run_id: str
    status: str
    final_metrics: Dict[str, float]


# --- Stage 4: REVIEW / EDIT ---

class ReportReadyPayload(BaseModel):
    asset_url: str


class NotebookReadyPayload(BaseModel):
    asset_url: str


class CodeWorkspaceReadyPayload(BaseModel):
    files: List[FileInfo]


class EditSuggestionsPayload(BaseModel):
    suggestions: List[Dict[str, Any]]


# --- Stage 5: EXPORT ---

class ExportProgressPayload(BaseModel):
    pct: float
    message: str


class ExportReadyPayload(BaseModel):
    asset_url: str
    contents: List[str]
    checksum: str


# ============================================================================
# STATE SNAPSHOT
# ============================================================================

class StateSnapshot(BaseModel):
    """Complete project state snapshot"""
    project_id: str
    stage: Stage
    decisions: Dict[str, Any]
    plans: Dict[str, Any]
    artifacts: List[ArtifactInfo]
    files: List[FileInfo]
    ui_layout: Dict[str, Any]
