"""
Assets API - Placeholder for file uploads and asset management.
Will be implemented in Task 3.2.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/assets", tags=["assets"])


@router.get("/")
async def list_assets():
    """Placeholder - list assets."""
    return {"assets": []}