from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os

router = APIRouter()

ASSETS_DIR = "data/assets"

@router.post("/assets/upload", response_model=dict)
async def upload_asset(file: UploadFile = File(...)):
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    
    file_location = os.path.join(ASSETS_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())
    
    return {"filename": file.filename, "url": f"/assets/{file.filename}"}

@router.get("/assets/{asset_id}", response_model=dict)
async def get_asset(asset_id: str):
    file_path = os.path.join(ASSETS_DIR, asset_id)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Asset not found")
    
    return {"url": f"/assets/{asset_id}"}