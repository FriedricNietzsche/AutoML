"""
Data ingestion and profiling endpoints.
- Upload CSV
- Download from Kaggle
- Emit dataset sample + profiling events to the WS bus
"""
import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Callable

import logging
import pandas as pd
from fastapi import APIRouter, Body, HTTPException, UploadFile, File
from PIL import Image
from huggingface_hub import HfApi

from app.events.bus import event_bus
from app.events.schema import (
    EventType,
    StageID,
    StageStatus,
)
from app.orchestrator.conductor import conductor
from app.orchestrator.pipeline import orchestrator as pipeline_orchestrator
from app.api.assets import ASSET_ROOT

router = APIRouter(prefix="/api/projects", tags=["data"])
log = logging.getLogger(__name__)


def _project_dir(project_id: str) -> Path:
    path = ASSET_ROOT / "projects" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _asset_rel(path: Path) -> str:
    return str(path.relative_to(ASSET_ROOT))


def _asset_url(path: Path) -> str:
    return f"/api/assets/{_asset_rel(path)}"


def _hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


async def _emit_sample(project_id: str, df: pd.DataFrame, sample_path: Path):
    payload = {
        "asset_url": _asset_url(sample_path),
        "columns": list(df.columns),
        "n_rows": len(df),
    }
    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.DATASET_SAMPLE_READY,
        payload=payload,
        stage_id=StageID.DATA_SOURCE,
        stage_status=StageStatus.IN_PROGRESS,
    )


async def _emit_profile(project_id: str, df: pd.DataFrame, profile_dir: Path):
    n_rows, n_cols = df.shape
    missing_pct = float(df.isna().mean().mean() * 100)
    types_breakdown = df.dtypes.astype(str).value_counts().to_dict()
    warnings: List[str] = []
    if missing_pct > 5:
        warnings.append("Dataset has >5% missing values.")

    profile = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_pct": missing_pct,
        "types_breakdown": types_breakdown,
        "warnings": warnings,
    }

    # Missingness table
    missingness = df.isna().mean().to_dict()
    missing_path = profile_dir / "missingness.json"
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.write_text(pd.Series(missingness).to_json(), encoding="utf-8")

    # Target distribution: pick a hashable column (prefer label/target)
    target_col = None
    for cand in ["target", "label"]:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        for col in df.columns:
            try:
                df[col].value_counts()
                target_col = col
                break
            except Exception:
                continue
    if target_col:
        try:
            target_counts = df[target_col].value_counts(normalize=True).to_dict()
            target_path = profile_dir / "target_distribution.json"
            target_path.write_text(pd.Series(target_counts).to_json(), encoding="utf-8")
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.TARGET_DISTRIBUTION_READY,
                payload={"asset_url": _asset_url(target_path)},
                stage_id=StageID.PROFILE_DATA,
                stage_status=StageStatus.IN_PROGRESS,
            )
        except Exception:
            pass

    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.PROFILE_SUMMARY,
        payload=profile,
        stage_id=StageID.PROFILE_DATA,
        stage_status=StageStatus.IN_PROGRESS,
    )
    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.MISSINGNESS_TABLE_READY,
        payload={"asset_url": _asset_url(missing_path)},
        stage_id=StageID.PROFILE_DATA,
        stage_status=StageStatus.IN_PROGRESS,
    )

    # Simple preprocess plan
    preprocess = {
        "steps": [
          {"name": "impute_numeric", "strategy": "mean"},
          {"name": "impute_categorical", "strategy": "most_frequent"},
          {"name": "scale_numeric", "strategy": "standard"},
        ]
    }
    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.PREPROCESS_PLAN,
        payload=preprocess,
        stage_id=StageID.PREPROCESS,
        stage_status=StageStatus.PENDING,
    )


async def _emit_waiting_confirmation(project_id: str, summary: str, candidates: List[str]):
    await event_bus.publish_event(
        project_id=project_id,
        event_name=EventType.WAITING_CONFIRMATION,
        payload={"stage_id": StageID.DATA_SOURCE.value, "summary": summary, "next_actions": candidates},
        stage_id=StageID.DATA_SOURCE,
        stage_status=StageStatus.WAITING_CONFIRMATION,
    )
    await conductor.transition_to(project_id, StageID.DATA_SOURCE, StageStatus.WAITING_CONFIRMATION, summary)


@router.post("/{project_id}/upload")
async def upload_dataset(project_id: str, file: UploadFile = File(...)):
    """Upload a dataset file and store selection in orchestrator context"""
    from ..orchestrator.pipeline import orchestrator
    
    project_dir = _project_dir(project_id)
    dest = project_dir / file.filename
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    try:
        df = pd.read_csv(dest, nrows=500)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    sample_path = project_dir / "sample.csv"
    df.head(200).to_csv(sample_path, index=False)

    # Store the uploaded dataset in orchestrator context
    async with orchestrator._lock:
        context = orchestrator._get_context(project_id)
        context["selected_dataset"] = {
            "source": "upload",
            "filename": file.filename,
            "path": str(dest),
            "rows": len(df),
            "columns": list(df.columns)
        }

    # Emit events
    await _emit_sample(project_id, df, sample_path)
    
    # Note: Don't auto-advance stages here - let user confirm first
    # The orchestrator will handle stage transitions when user clicks confirm

    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}


@router.post("/{project_id}/dataset/select")
async def select_dataset(project_id: str, dataset_id: str = Body(..., embed=True)):
    """
    User selects a dataset from the suggested candidates.
    Downloads it from HuggingFace and stores path in orchestrator context.
    """
    print(f"[API] User selected dataset: {dataset_id} for project: {project_id}")
    
    async with pipeline_orchestrator._lock:
        context = pipeline_orchestrator._get_context(project_id)
        # Find the selected dataset from candidates
        candidates = context.get("dataset_candidates", [])
        selected = next((d for d in candidates if d.get("id") == dataset_id), None)
        
        if not selected:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found in candidates")
        
        # Publish download start event
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload={
                "stage": StageID.DATA_SOURCE.value,
                "status": StageStatus.IN_PROGRESS.value,
                "message": f"⬇️ Downloading dataset: {selected.get('name', dataset_id)}...",
                "details": "This may take 30-60 seconds for large datasets"
            },
            stage_id=StageID.DATA_SOURCE,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        # Download dataset from HuggingFace
        print(f"[API] Downloading dataset from HuggingFace: {dataset_id}")
        try:
            from datasets import load_dataset
            import pandas as pd
            
            # Publish fetching event
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.STAGE_STATUS,
                payload={
                    "stage": StageID.DATA_SOURCE.value,
                    "status": StageStatus.IN_PROGRESS.value,
                    "message": f"📦 Fetching dataset metadata from HuggingFace...",
                },
                stage_id=StageID.DATA_SOURCE,
                stage_status=StageStatus.IN_PROGRESS,
            )
            
            # Load from HuggingFace
            dataset = load_dataset(dataset_id, split='train[:1000]')  # Limit to 1000 samples for speed
            
            # Publish processing event
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.STAGE_STATUS,
                payload={
                    "stage": StageID.DATA_SOURCE.value,
                    "status": StageStatus.IN_PROGRESS.value,
                    "message": f"🔄 Processing dataset (1000 samples)...",
                },
                stage_id=StageID.DATA_SOURCE,
                stage_status=StageStatus.IN_PROGRESS,
            )
            
            # Convert to pandas DataFrame
            df = dataset.to_pandas()
            
            # Save to project directory
            project_dir = _project_dir(project_id)
            dataset_path = project_dir / f"{dataset_id.replace('/', '_')}.csv"
            df.to_csv(dataset_path, index=False)
            
            print(f"[API] ✅ Downloaded and saved {len(df)} rows to {dataset_path}")
            
            # Store selection with path
            selected["path"] = str(dataset_path)
            selected["rows"] = len(df)
            selected["columns"] = list(df.columns)
            context["selected_dataset"] = selected
            
            # Publish success event
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.DATASET_SELECTED,
                payload={
                    "dataset": selected,
                    "message": f"✅ Dataset downloaded: {len(df)} rows, {len(df.columns)} columns"
                },
                stage_id=StageID.DATA_SOURCE,
                stage_status=StageStatus.IN_PROGRESS,
            )
            
            return {"status": "ok", "selected": selected, "path": str(dataset_path)}
            
        except Exception as e:
            print(f"[API] ❌ Error downloading dataset: {e}")
            
            # Publish error event
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.STAGE_STATUS,
                payload={
                    "stage": StageID.DATA_SOURCE.value,
                    "status": StageStatus.ERROR.value,
                    "message": f"❌ Failed to download dataset: {str(e)}",
                },
                stage_id=StageID.DATA_SOURCE,
                stage_status=StageStatus.ERROR,
            )
            
            # Still store selection but without path - will use mock data
            context["selected_dataset"] = selected
            raise HTTPException(status_code=500, detail=f"Failed to download dataset: {e}")


@router.post("/{project_id}/model/select")
async def select_model(project_id: str, model_id: str = Body(..., embed=True)):
    """
    User selects a model from the recommended candidates.
    Stores the selection in orchestrator context.
    """
    print(f"[API] User selected model: {model_id} for project: {project_id}")
    
    async with pipeline_orchestrator._lock:
        context = pipeline_orchestrator._get_context(project_id)
        # Find the selected model from candidates
        candidates = context.get("model_candidates", [])
        selected = next((m for m in candidates if m.get("id") == model_id), None)
        
        if not selected:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found in candidates")
        
        # Store selection
        context["selected_model"] = selected
        print(f"[API] ✅ Selected model: {selected.get('name', model_id)}")
        
        # Publish event
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload={
                "stage": StageID.MODEL_SELECT.value,
                "status": StageStatus.IN_PROGRESS.value,
                "message": f"✅ Selected model: {selected.get('name', model_id)}",
                "model": selected
            },
            stage_id=StageID.MODEL_SELECT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        return {"status": "ok", "selected": selected}


@router.get("/hf/search")
async def search_hf_datasets(query: str, limit: int = 20):
    """
    Search Hugging Face datasets by keyword and return only public, allowed-license entries.
    """
    token = _hf_token()
    api = HfApi(token=token)
    allowed_licenses = {"mit", "apache-2.0", "cc-by-4.0", "cc0-1.0", "cc-by-sa-4.0", "unlicense"}
    results = []
    try:
        for ds in api.list_datasets(search=query, limit=limit):
            # Skip private/gated
            if getattr(ds, "private", False) or getattr(ds, "gated", False):
                continue
            info = ds.card_data or {}
            lic_raw = (info.get("license") or "").lower()
            lic = lic_raw.strip()
            if lic and lic not in allowed_licenses:
                continue
            results.append({"id": ds.id, "license": lic or "unspecified"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF search failed: {e}")
    if not results:
        return {
            "results": [],
            "count": 0,
            "message": "No public datasets with allowed licenses found for this query. Try a different keyword or provide a dataset link.",
        }
    return {"results": results, "count": len(results)}


@router.post("/{project_id}/ingest/hf")
async def ingest_hf(
    project_id: str,
    dataset: str = Body(..., embed=True),
    split: str = Body("train", embed=True),
    max_rows: int = Body(500, embed=True),
):
    """
    Ingest a dataset from Hugging Face Hub using `datasets`.
    Suitable for public datasets (e.g., mnist). Requires `datasets` package.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"datasets package not available: {e}")

    project_dir = _project_dir(project_id)
    log.info("Downloading HF dataset %s split=%s (max_rows=%s)...", dataset, split, max_rows)
    try:
        token = _hf_token()
        ds = load_dataset(dataset, split=split, token=token)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset {dataset}:{split} from HF: {e}")

    try:
        import numpy as np  # local import for hf conversion
        df = ds.to_pandas()
        if "image" in df.columns:
            imgs = df["image"].apply(lambda x: np.array(x).reshape(-1).tolist())
            pix_cols = [f"pix_{i}" for i in range(len(imgs.iloc[0]))]
            pix_df = pd.DataFrame(imgs.tolist(), columns=pix_cols)
            df = pd.concat([pix_df, df.drop(columns=["image"])], axis=1)
    except Exception:
        try:
            df = ds.to_dataframe()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to convert dataset to DataFrame: {e}")

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    dest = project_dir / f"{dataset.replace('/', '_')}_{split}.csv"
    df.to_csv(dest, index=False)

    sample_path = project_dir / "sample.csv"
    df.head(min(200, len(df))).to_csv(sample_path, index=False)

    await _emit_sample(project_id, df, sample_path)
    await conductor.transition_to(project_id, StageID.DATA_SOURCE, StageStatus.COMPLETED, "HF dataset ingested")
    await conductor.transition_to(project_id, StageID.PROFILE_DATA, StageStatus.IN_PROGRESS, "Profiling dataset")
    await _emit_profile(project_id, df, project_dir / "profile")

    return {"status": "ok", "rows": len(df), "columns": list(df.columns)}


@router.post("/{project_id}/ingest/hf-images")
async def ingest_hf_images(
    project_id: str,
    dataset: str = Body(..., embed=True),
    split: str = Body("train", embed=True),
    image_field: str = Body("image", embed=True),
    label_field: str = Body("label", embed=True),
    max_images: int = Body(40, embed=True),
):
    """
    Download a small image dataset from Hugging Face and save to images/ subdir for vision training.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"datasets package not available: {e}")

    token = _hf_token()
    streaming = True
    try:
        ds = load_dataset(dataset, split=split, streaming=True, token=token)
    except Exception:
        streaming = False
        try:
            ds = load_dataset(dataset, split=split, token=token)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load dataset {dataset}:{split} from HF: {e}")

    project_dir = _project_dir(project_id)
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    label_map = {}
    iterator = ds if streaming else iter(ds)
    for row in iterator:
        if saved >= max_images:
            break
        img = row[image_field] if isinstance(row, dict) else getattr(row, image_field, None)
        label = row.get(label_field, "unknown") if isinstance(row, dict) else getattr(row, label_field, "unknown")
        try:
            pil_img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        except Exception:
            continue
        fname = images_dir / f"{label}_{saved}.png"
        pil_img.save(fname)
        label_map[str(fname.name)] = label
        saved += 1

    if saved == 0:
        raise HTTPException(status_code=400, detail="No images saved from HF dataset")

    # Save labels map
    labels_path = project_dir / "images" / "labels.json"
    import json
    labels_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    # Emit sample and basic profile
    df = pd.DataFrame({"file": list(label_map.keys()), "label": list(label_map.values())})
    sample_path = project_dir / "sample.csv"
    df.to_csv(sample_path, index=False)

    await _emit_sample(project_id, df, sample_path)
    await conductor.transition_to(project_id, StageID.DATA_SOURCE, StageStatus.COMPLETED, "HF images ingested")
    await conductor.transition_to(project_id, StageID.PROFILE_DATA, StageStatus.IN_PROGRESS, "Profiling dataset")
    await _emit_profile(project_id, df, project_dir / "profile")

    return {"status": "ok", "images": saved, "labels": list(set(label_map.values()))}


@router.post("/{project_id}/ingest/kaggle")
async def ingest_kaggle(project_id: str, dataset: str = Body(..., embed=True), file_name: Optional[str] = Body(None, embed=True)):
    """Download a Kaggle dataset (requires KAGGLE_USERNAME/KAGGLE_KEY env)."""
    import kaggle  # type: ignore

    project_dir = _project_dir(project_id)
    cache_dir = ASSET_ROOT / "cache" / "kaggle" / dataset
    tmpdir = tempfile.mkdtemp()

    # If cached, reuse without network
    if cache_dir.exists():
        candidates = list(cache_dir.glob("*.csv"))
        if file_name:
            candidates = [c for c in candidates if c.name == file_name]
        if candidates:
            log.info("Using cached Kaggle dataset %s (%s)", dataset, candidates[0].name)
            chosen = candidates[0]
            dest = project_dir / chosen.name
            shutil.copy2(chosen, dest)
            df = pd.read_csv(dest, nrows=500)
            sample_path = project_dir / "sample.csv"
            df.head(200).to_csv(sample_path, index=False)
            await _emit_sample(project_id, df, sample_path)
            await conductor.transition_to(project_id, StageID.DATA_SOURCE, StageStatus.COMPLETED, "Kaggle dataset ingested (cache)")
            await conductor.transition_to(project_id, StageID.PROFILE_DATA, StageStatus.IN_PROGRESS, "Profiling dataset")
            await _emit_profile(project_id, df, project_dir / "profile")
            return {"status": "ok", "rows": len(df), "columns": list(df.columns)}

    try:
        kaggle.api.authenticate()
        log.info("Downloading Kaggle dataset %s ...", dataset)
        kaggle.api.dataset_download_files(dataset, path=tmpdir, unzip=True, quiet=True)

        chosen = None
        for root, _, files in os.walk(tmpdir):
            for name in files:
                if file_name and name != file_name:
                    continue
                if name.lower().endswith(".csv"):
                    chosen = Path(root) / name
                    break
            if chosen:
                break
        if not chosen:
            raise HTTPException(status_code=400, detail="No CSV file found in dataset")

        dest = project_dir / chosen.name
        shutil.move(str(chosen), dest)
        # Cache for reuse
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, cache_dir / dest.name)

        df = pd.read_csv(dest, nrows=500)
        sample_path = project_dir / "sample.csv"
        df.head(200).to_csv(sample_path, index=False)

        await _emit_sample(project_id, df, sample_path)
        await conductor.transition_to(project_id, StageID.DATA_SOURCE, StageStatus.COMPLETED, "Kaggle dataset ingested")
        await conductor.transition_to(project_id, StageID.PROFILE_DATA, StageStatus.IN_PROGRESS, "Profiling dataset")
        await _emit_profile(project_id, df, project_dir / "profile")

        return {"status": "ok", "rows": len(df), "columns": list(df.columns)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.post("/{project_id}/ingest/auto")
async def ingest_auto(project_id: str, dataset_hint: str = Body(..., embed=True), file_name: Optional[str] = Body(None, embed=True)):
    """
    Auto-ingest from Kaggle using dataset_hint. If multiple matches, emit WAITING_CONFIRMATION.
    """
    import kaggle  # type: ignore

    if not dataset_hint.strip():
        raise HTTPException(status_code=400, detail="dataset_hint is required")

    try:
        kaggle.api.authenticate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kaggle auth failed: {e}")

    def is_public(ref: str) -> bool:
        try:
            files = kaggle.api.dataset_list_files(ref)
            # If we can list files without auth error, treat as public.
            return bool(files) or files is not None
        except Exception:
            return False

    # If the hint looks like an explicit slug, short-circuit to that.
    if "/" in dataset_hint:
        refs = [dataset_hint]
    else:
        matches = kaggle.api.dataset_list(search=dataset_hint)
        refs = [d.ref for d in matches]
    # Curated public shortcuts for demos (choose public/gated-free datasets)
    curated = {
        "catsdogs": "shaunthesheep/microsoft-catsvsdogs-dataset",  # adjust if gated
        "cats vs dogs": "shaunthesheep/microsoft-catsvsdogs-dataset",
        "alzheimer": "tawsifurrahman/alzheimers-dataset-4-class-of-images",
    }
    if dataset_hint.lower() in curated:
        refs = [curated[dataset_hint.lower()]]

    public_refs = [ref for ref in refs if is_public(ref)]

    if not public_refs:
        raise HTTPException(status_code=404, detail="No public Kaggle datasets found for hint; please provide a dataset or upload.")
    if len(public_refs) > 1:
        await _emit_waiting_confirmation(project_id, f"Multiple public datasets found for '{dataset_hint}'", public_refs[:5])
        return {"status": "needs_confirmation", "candidates": public_refs[:5]}

    chosen = public_refs[0]
    # If caller provided a specific file_name, pass it through; otherwise let ingest_kaggle pick first CSV.
    return await ingest_kaggle(project_id, dataset=chosen, file_name=file_name)
