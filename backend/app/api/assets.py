"""
Assets API - minimal local-disk asset serving scaffold.
Stores under ASSET_ROOT (defaults to ./data/assets) and serves files by relative path.
"""
import os
from pathlib import Path
from typing import List

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
async def list_assets() -> dict:
    assets: List[str] = []
    for root, _, files in os.walk(ASSET_ROOT):
        for name in files:
            full = Path(root) / name
            rel = full.relative_to(ASSET_ROOT)
            assets.append(str(rel))
    return {"assets": assets}


@router.post("/upload")
async def upload_asset(file: UploadFile = File(...)) -> dict:
    dest = _safe_path(file.filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return {"asset": str(dest.relative_to(ASSET_ROOT))}


@router.get("/{asset_path:path}")
async def get_asset(asset_path: str):
    path = _safe_path(asset_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(path)
