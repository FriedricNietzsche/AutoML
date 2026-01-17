from pydantic import BaseModel
from typing import Any, Dict, List, Union

class EventPayload(BaseModel):
    name: str
    payload: Dict[str, Any]

class Stage(BaseModel):
    id: str
    index: int
    status: str

class EventMessage(BaseModel):
    v: int
    type: str
    project_id: str
    seq: int
    ts: int
    stage: Stage
    event: EventPayload

class StateSnapshot(BaseModel):
    project_id: str
    stage: Stage
    decisions: Dict[str, Any]
    plans: Dict[str, Any]
    artifacts: List[Dict[str, Union[str, Any]]]
    files: List[Dict[str, Union[str, int]]]
    ui_layout: Dict[str, Any]