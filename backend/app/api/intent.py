"""
Endpoints for intent parsing and model selection (Stage PARSE_INTENT / MODEL_SELECT).
Uses OpenRouter-backed prompt parser with a defensive fallback.
"""
from fastapi import APIRouter, Body, HTTPException

from app.agents.prompt_parser import PromptParserAgent
from app.agents.model_selector import ModelSelectorAgent
from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus
from app.orchestrator.conductor import conductor

router = APIRouter(prefix="/api/projects", tags=["intent"])


@router.post("/{project_id}/parse")
async def parse_intent(project_id: str, prompt: str = Body(..., embed=True)):
    agent = PromptParserAgent()
    parsed = agent.parse(prompt)

    # Emit PROMPT_PARSED event
    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.PROMPT_PARSED,
        payload=parsed,
        stage_id=StageID.PARSE_INTENT,
        stage_status=StageStatus.IN_PROGRESS,
    )

    # Mark PARSE_INTENT completed and advance to DATA_SOURCE
    await conductor.transition_to(project_id, StageID.PARSE_INTENT, StageStatus.COMPLETED, "Intent parsed")
    await conductor.transition_to(project_id, StageID.DATA_SOURCE, StageStatus.IN_PROGRESS, "Awaiting data source")

    return {"status": "parsed", "payload": parsed}


@router.post("/{project_id}/model/select")
async def select_model(project_id: str, task_type: str = Body(..., embed=True)):
    selector = ModelSelectorAgent()
    try:
        models = selector.select_model(task_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    payload = {"models": models}
    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.MODEL_CANDIDATES,
        payload=payload,
        stage_id=StageID.MODEL_SELECT,
        stage_status=StageStatus.IN_PROGRESS,
    )

    return {"models": models}


"""
Intent API - Prompt parsing and model selection.
Placeholder - will be implemented in Phase 3.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/intent", tags=["intent"])


@router.post("/parse/{project_id}")
async def parse_prompt(project_id: str, prompt: str = ""):
    """
    Parse user prompt to extract intent.
    Placeholder - returns mock data.
    """
    return {
        "project_id": project_id,
        "task_type": "classification",
        "target": "target",
        "constraints": [],
        "parsed": True,
    }


@router.get("/models/{project_id}")
async def get_model_candidates(project_id: str):
    """
    Get candidate models for the task.
    Placeholder.
    """
    return {
        "models": [
            {"id": "random_forest", "name": "Random Forest", "family": "ensemble"},
            {"id": "logistic_regression", "name": "Logistic Regression", "family": "linear"},
            {"id": "gradient_boosting", "name": "Gradient Boosting", "family": "ensemble"},
        ]
    }
