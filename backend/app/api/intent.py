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
