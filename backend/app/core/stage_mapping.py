"""
Stage mapping: 8 backend stages → 5 UI buckets

Backend stages:
  PARSE, DATA_SOURCE, PROFILE_DATA, PREPROCESS, 
  MODEL_SELECT, TRAIN, REVIEW, EXPORT

UI buckets (indices 0-4):
  0: PARSE + DATA_SOURCE
  1: PROFILE_DATA
  2: PREPROCESS
  3: MODEL_SELECT + TRAIN
  4: REVIEW + EXPORT
"""

from enum import Enum
from typing import Dict, Any

class BackendStage(str, Enum):
    PARSE = "PARSE"
    DATA_SOURCE = "DATA_SOURCE"
    PROFILE_DATA = "PROFILE_DATA"
    PREPROCESS = "PREPROCESS"
    MODEL_SELECT = "MODEL_SELECT"
    TRAIN = "TRAIN"
    REVIEW = "REVIEW"
    EXPORT = "EXPORT"

# Map backend stage to UI bucket index
STAGE_TO_UI_BUCKET: Dict[str, int] = {
    "PARSE": 0,
    "DATA_SOURCE": 0,
    "PROFILE_DATA": 1,
    "PREPROCESS": 2,
    "MODEL_SELECT": 3,
    "TRAIN": 3,
    "REVIEW": 4,
    "EXPORT": 4,
}

# UI bucket labels for display
UI_BUCKET_LABELS = [
    "Data Ingestion",
    "Data Profiling", 
    "Preprocessing",
    "Model Training",
    "Review & Export",
]

def get_ui_bucket(stage: str) -> int:
    """Get UI bucket index for a backend stage."""
    return STAGE_TO_UI_BUCKET.get(stage.upper(), 0)

def get_bucket_label(bucket_idx: int) -> str:
    """Get display label for UI bucket."""
    if 0 <= bucket_idx < len(UI_BUCKET_LABELS):
        return UI_BUCKET_LABELS[bucket_idx]
    return "Unknown"

def build_stage_event(
    stage: str,
    status: str,
    message: str = "",
    progress: float = 0.0,
    artifacts: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Build a standardized STAGE_STATUS event payload."""
    return {
        "type": "STAGE_STATUS",
        "stage": stage.upper(),
        "uiBucket": get_ui_bucket(stage),
        "bucketLabel": get_bucket_label(get_ui_bucket(stage)),
        "status": status,  # "running" | "completed" | "error"
        "message": message,
        "progress": progress,
        "artifacts": artifacts or {},
    }
