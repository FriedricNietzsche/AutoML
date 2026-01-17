from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

router = APIRouter()

# In-memory storage for stages (for demonstration purposes)
stages = {
    "1": {"id": "PARSE_INTENT", "status": "PENDING"},
    "2": {"id": "DATA_SOURCE", "status": "PENDING"},
    "3": {"id": "PROFILE_DATA", "status": "PENDING"},
    "4": {"id": "PREPROCESS", "status": "PENDING"},
    "5": {"id": "MODEL_SELECT", "status": "PENDING"},
    "6": {"id": "TRAIN", "status": "PENDING"},
    "7": {"id": "REVIEW_EDIT", "status": "PENDING"},
    "8": {"id": "EXPORT", "status": "PENDING"},
}

@router.get("/stages", response_model=List[Dict[str, Any]])
async def get_stages():
    return list(stages.values())

@router.post("/stages/{stage_id}/confirm")
async def confirm_stage(stage_id: str):
    if stage_id not in stages:
        raise HTTPException(status_code=404, detail="Stage not found")
    
    stages[stage_id]["status"] = "CONFIRMED"
    return {"message": f"Stage {stage_id} confirmed", "stage": stages[stage_id]}