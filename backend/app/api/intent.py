"""
Endpoints for intent parsing and model selection (Stage PARSE_INTENT / MODEL_SELECT).
Uses the new PipelineOrchestrator to automatically handle all stages.
"""
from fastapi import APIRouter, Body, HTTPException

from app.orchestrator.pipeline import orchestrator

router = APIRouter(prefix="/api/projects", tags=["intent"])


@router.post("/{project_id}/parse")
async def parse_intent(project_id: str, prompt: str = Body(..., embed=True)):
    """
    Start the AutoML pipeline with a user prompt.
    The orchestrator will automatically handle all stages.
    """
    print(f"\n{'='*60}")
    print(f"[API] Starting pipeline for project: {project_id}")
    print(f"[API] Prompt: {prompt[:100]}...")
    print(f"{'='*60}\n")
    
    result = await orchestrator.start_pipeline(project_id, prompt)
    
    print(f"\n{'='*60}")
    print(f"[API] Pipeline started successfully")
    print(f"{'='*60}\n")
    
    return result
