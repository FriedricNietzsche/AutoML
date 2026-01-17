from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

router = APIRouter()

# In-memory storage for projects (for demonstration purposes)
projects_db = {}

@router.post("/api/projects/", response_model=Dict[str, Any])
async def create_project(project: Dict[str, Any]):
    project_id = len(projects_db) + 1
    projects_db[project_id] = project
    return {"project_id": project_id, **project}

@router.get("/api/projects/{project_id}", response_model=Dict[str, Any])
async def get_project(project_id: int):
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project_id": project_id, **project}

@router.put("/api/projects/{project_id}", response_model=Dict[str, Any])
async def update_project(project_id: int, project: Dict[str, Any]):
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    projects_db[project_id] = project
    return {"project_id": project_id, **project}

@router.delete("/api/projects/{project_id}", response_model=Dict[str, Any])
async def delete_project(project_id: int):
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    del projects_db[project_id]
    return {"detail": "Project deleted successfully"}