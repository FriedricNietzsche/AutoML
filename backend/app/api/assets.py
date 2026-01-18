"""
Assets API - File and artifact management.
"""
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

ASSET_ROOT = Path(os.getenv("ASSET_ROOT", "data/assets")).resolve()
ASSET_ROOT.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api/assets", tags=["assets"])


def _safe_path(rel_path: str) -> Path:
    path = (ASSET_ROOT / rel_path).resolve()
    if not str(path).startswith(str(ASSET_ROOT)):
        raise HTTPException(status_code=400, detail="Invalid asset path")
    return path


@router.get("/")
async def list_assets(project_id: Optional[str] = None):
    """List all assets, optionally filtered by project."""
    assets: List[str] = []
    for root, _, files in os.walk(ASSET_ROOT):
        for name in files:
            full = Path(root) / name
            rel = full.relative_to(ASSET_ROOT)
            assets.append(str(rel))
    return {"assets": assets, "project_id": project_id}


@router.post("/upload")
async def upload_asset(file: UploadFile = File(...)) -> dict:
    dest = _safe_path(file.filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return {"asset": str(dest.relative_to(ASSET_ROOT))}


@router.get("/{asset_id:path}")
async def get_asset(asset_id: str):

    """Get asset details by ID."""
    path = _safe_path(asset_id)
    if not path.exists() or not path.is_file():
        return {
            "asset_id": asset_id,
            "status": "not_found",
            "message": "Asset management not yet implemented",
        }
    return FileResponse(path)
