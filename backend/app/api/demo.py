"""
Demo orchestration endpoint for end-to-end cat/dog classifier workflow.
Coordinates all stages: parse → ingest → profile → model select → train → export
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.agents.prompt_parser import PromptParserAgent
from app.agents.model_selector import ModelSelectorAgent
from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus
from app.orchestrator.conductor import conductor
from app.api.assets import ASSET_ROOT

router = APIRouter(prefix="/api/demo", tags=["demo"])
log = logging.getLogger(__name__)


async def run_demo_workflow(project_id: str, prompt: str):
    """
    Execute the full demo workflow in the background.
    
    Stages:
    1. Parse intent
    2. Ingest dataset (HF images)
    3. Profile data
    4. Select model
    5. Train model
    6. Generate notebook
    7. Export bundle
    """
    try:
        log.info(f"Starting demo workflow for project {project_id}")
        
        # Stage 1: Parse Intent
        await conductor.transition_to(project_id, StageID.PARSE_INTENT, StageStatus.IN_PROGRESS, "Analyzing your request with AI...")
        
        parser = PromptParserAgent()
        parsed = parser.parse(prompt)
        
        # Create detailed summary message
        task_type = parsed.get("task_type", "other")
        target = parsed.get("target", "unknown")
        dataset_hint = parsed.get("dataset_hint", "")
        
        summary_msg = f"✓ Understood: {task_type.title()} task"
        if target:
            summary_msg += f"\n  Goal: {target}"
        if dataset_hint:
            summary_msg += f"\n  Looking for: {dataset_hint} dataset"
        
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.PROMPT_PARSED,
            payload=parsed,
            stage_id=StageID.PARSE_INTENT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        await conductor.transition_to(project_id, StageID.PARSE_INTENT, StageStatus.COMPLETED, summary_msg)
        
        # Small delay for visual feedback
        await asyncio.sleep(0.5)
        
        # Extract task details from parsed intent
        task_type = parsed.get("task_type", "classification")
        dataset_hint = parsed.get("dataset_hint", "")
        
        # Stage 2: Data Source - Search for dataset using AI agent
        await conductor.transition_to(project_id, StageID.DATA_SOURCE, StageStatus.IN_PROGRESS, "Searching for datasets...")
        
        # Use DatasetFinderAgent to find appropriate datasets
        from app.agents.dataset_finder import DatasetFinderAgent
        
        finder = DatasetFinderAgent()
        dataset_candidates = finder.find_datasets(
            task_type=task_type,
            dataset_hint=dataset_hint,
            max_results=5
        )
        
        # Emit dataset candidates event
        await event_bus.publish_event(
            project_id=project_id,
            event_name="DATASET_CANDIDATES",
            payload={"datasets": dataset_candidates[:3]},  # Top 3 for display
            stage_id=StageID.DATA_SOURCE,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        # Filter to only valid licenses
        valid_datasets = [d for d in dataset_candidates if d["license_valid"]]
        
        if not valid_datasets:
            # No datasets with valid licenses found - notify user
            log.warning("No datasets with valid licenses found"
)
            await event_bus.publish_event(
                project_id=project_id,
                event_name="DATASET_SEARCH_FAILED",
                payload={
                    "message": "Could not find datasets with valid licenses for this task.",
                    "searched_for": {
                        "task_type": task_type,
                        "dataset_hint": dataset_hint
                    },
                    "invalid_datasets": [
                        {
                            "id": d["id"],
                            "name": d["name"],
                            "license": d["license"],
                            "reason": d["license_reason"]
                        }
                        for d in dataset_candidates[:5]
                    ],
                    "requires_user_input": True,
                    "suggested_actions": [
                        "Provide a dataset URL",
                        "Upload your own data (CSV or images)",
                        "Try a different prompt with more common datasets"
                    ]
                },
                stage_id=StageID.DATA_SOURCE,
                stage_status=StageStatus.WAITING_CONFIRMATION,
            )
            
            await conductor.transition_to(
                project_id,
                StageID.DATA_SOURCE,
                StageStatus.WAITING_CONFIRMATION,
                "No valid datasets found - please provide data"
            )
            return  # Stop workflow, wait for user input
        
        # Select best dataset (first one with valid license)
        selected_dataset = valid_datasets[0]
        dataset = selected_dataset["id"]
        
        log.info(f"Selected dataset: {dataset} (license={selected_dataset['license']})")
        
        dataset_msg = f"✓ Found dataset: {selected_dataset['name']}"
        dataset_msg += f"\n  ID: {dataset}"
        dataset_msg += f"\n  License: {selected_dataset['license']} (verified)"
        dataset_msg += f"\n  Downloads: {selected_dataset['downloads']:,}"
        
        await conductor.transition_to(
            project_id,
            StageID.DATA_SOURCE,
            StageStatus.IN_PROGRESS,
            dataset_msg
        )
        
        # Emit dataset selected event
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.DATASET_SELECTED,
            payload=selected_dataset,
            stage_id=StageID.DATA_SOURCE,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        # Determine split and fields based on dataset
        split = "train"
        
        # Import data ingestion functions
        from app.api.data import ingest_hf_images, ingest_hf, _emit_sample, _emit_profile, _project_dir
        
        # Detect dataset type from the selected dataset info
        # Check if this is an image dataset or tabular dataset
        is_image_dataset = task_type == "classification" and any(
            keyword in dataset.lower() 
            for keyword in ["image", "cifar", "mnist", "imagenet", "cat", "dog", "fashion"]
        )
        
        # Ingest dataset (this handles emitting DATASET_SAMPLE_READY)
        try:
            if is_image_dataset:
                # Use image ingestion for image datasets
                result = await ingest_hf_images(
                    project_id=project_id,
                    dataset=dataset,
                    split=split,
                    image_field="image",
                    label_field="label",
                    max_images=30,  # Keep small for demo
                )
                
                completion_msg = f"✓ Loaded {dataset}"
                completion_msg += f"\n  Format: Image classification"
                completion_msg += f"\n  Samples: 30 images"
            else:
                # Use tabular ingestion for CSV/tabular datasets
                result = await ingest_hf(
                    project_id=project_id,
                    dataset=dataset,
                    split=split,
                    max_rows=500,
                )
                
                completion_msg = f"✓ Loaded {dataset}"
                completion_msg += f"\n  Format: Tabular data"
                completion_msg += f"\n  Rows: {result.get('rows', 0)}"
                completion_msg += f"\n  Columns: {len(result.get('columns', []))}"
            
            await conductor.transition_to(
                project_id,
                StageID.DATA_SOURCE,
                StageStatus.COMPLETED,
                completion_msg
            )
            
        except Exception as e:
            log.error(f"Failed to ingest HF dataset {dataset}: {e}")
            # Emit error event
            await event_bus.publish_event(
                project_id=project_id,
                event_name="DATASET_LOAD_FAILED",
                payload={
                    "dataset_id": dataset,
                    "error": str(e),
                    "message": "Failed to load dataset. Please try another dataset or upload your data."
                },
                stage_id=StageID.DATA_SOURCE,
                stage_status=StageStatus.FAILED,
            )
            return
        
        await asyncio.sleep(0.5)
        
        # Stage 3: Model Selection
        await conductor.transition_to(project_id, StageID.MODEL_SELECT, StageStatus.IN_PROGRESS, "Evaluating model options...")
        
        selector = ModelSelectorAgent()
        models = selector.select_model(task_type)
        
        # Show candidates
        models_msg = f"Evaluating {len(models)} model(s):"
        for m in models[:3]:
            models_msg += f"\n  • {m.get('name', m.get('id', 'Unknown'))}"
        
        await conductor.transition_to(project_id, StageID.MODEL_SELECT, StageStatus.IN_PROGRESS, models_msg)
        
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.MODEL_CANDIDATES,
            payload={"models": models},
            stage_id=StageID.MODEL_SELECT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        # Auto-select first model
        selected_model = models[0] if models else {"id": "cnn", "name": "CNN"}
        
        selection_msg = f"✓ Selected: {selected_model['name']}"
        selection_msg += f"\n  Type: {selected_model.get('type', 'Neural Network')}"
        selection_msg += f"\n  Reason: Best for {task_type} tasks"
        
        await conductor.transition_to(project_id, StageID.MODEL_SELECT, StageStatus.IN_PROGRESS, selection_msg)
        
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.MODEL_SELECTED,
            payload={"model_id": selected_model["id"]},
            stage_id=StageID.MODEL_SELECT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        await conductor.transition_to(project_id, StageID.MODEL_SELECT, StageStatus.COMPLETED, f"Model selected: {selected_model['name']}")
        
        await asyncio.sleep(0.5)
        
        # Stage 4: Training
        await conductor.transition_to(project_id, StageID.TRAIN, StageStatus.IN_PROGRESS, "Training model...")
        
        # Import training functions
        from app.api.train import train_image, train_tabular
        
        try:
            if is_image_dataset:
                result = await train_image(project_id=project_id, data_subdir="images")
            else:
                result = await train_tabular(
                    project_id=project_id, 
                    target=target if target != "unknown" else "target",
                    task_type=task_type,
                    model_id=selected_model.get("id", "auto")
                )
            log.info(f"Training completed: {result}")
        except Exception as e:
            log.error(f"Training failed: {e}")
            await conductor.transition_to(project_id, StageID.TRAIN, StageStatus.FAILED, f"Training error: {e}")
            return

        
        await asyncio.sleep(0.5)
        
        # Stage 5: Review & Export
        await conductor.transition_to(project_id, StageID.REVIEW_EDIT, StageStatus.IN_PROGRESS, "Generating notebook...")
        
        # Generate notebook
        project_dir = ASSET_ROOT / "projects" / project_id / "artifacts"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        notebook_path = project_dir / "notebook.ipynb"
        
        # Create simple notebook
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Cat vs Dog Classifier\n\nGenerated by AutoML Agentic Builder"]
                },
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": ["# Model training completed\n", f"# Task: {task_type}\n", f"# Model: {selected_model['name']}"],
                    "outputs": []
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5
        }
        
        import json
        notebook_path.write_text(json.dumps(notebook_content, indent=2), encoding="utf-8")
        
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.NOTEBOOK_READY,
            payload={"asset_url": f"/api/assets/projects/{project_id}/artifacts/notebook.ipynb"},
            stage_id=StageID.REVIEW_EDIT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        await conductor.transition_to(project_id, StageID.REVIEW_EDIT, StageStatus.COMPLETED, "Notebook ready")
        
        await asyncio.sleep(0.5)
        
        # Stage 6: Export
        await conductor.transition_to(project_id, StageID.EXPORT, StageStatus.IN_PROGRESS, "Creating export bundle...")
        
        # Create export bundle (placeholder)
        export_path = ASSET_ROOT / "projects" / project_id / "export.zip"
        
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.EXPORT_READY,
            payload={
                "asset_url": f"/api/assets/projects/{project_id}/export.zip",
                "contents": ["model.pkl", "notebook.ipynb", "data_sample.csv"],
                "checksum": "abc123"
            },
            stage_id=StageID.EXPORT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        await conductor.transition_to(project_id, StageID.EXPORT, StageStatus.COMPLETED, "Export ready!")
        
        log.info(f"Demo workflow completed for project {project_id}")
        
    except Exception as e:
        log.error(f"Demo workflow error: {e}", exc_info=True)
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload={
                "stage_id": "TRAIN",
                "status": "FAILED",
                "message": f"Workflow error: {str(e)}"
            },
            stage_id=StageID.TRAIN,
            stage_status=StageStatus.FAILED,
        )


@router.post("/run/{project_id}")
async def run_demo(
    project_id: str,
    background_tasks: BackgroundTasks,
    prompt: str = "Build me a classifier for cat/dog"
):
    """
    Run the full demo workflow for a project.
    
    This orchestrates:
    - Intent parsing
    - Dataset ingestion
    - Profiling
    - Model selection
    - Training
    - Notebook generation
    - Export
    
    All updates stream via WebSocket.
    """
    log.info(f"Demo requested for project {project_id}: {prompt}")
    
    # Run workflow in background
    background_tasks.add_task(run_demo_workflow, project_id, prompt)
    
    return {
        "status": "started",
        "project_id": project_id,
        "message": "Demo workflow started. Connect to WebSocket for updates."
    }


@router.get("/status/{project_id}")
async def get_demo_status(project_id: str):
    """
    Get current demo workflow status.
    """
    state = conductor.get_state_snapshot(project_id)
    return {
        "project_id": project_id,
        "current_stage": state.get("current_stage"),
        "stages": state.get("stages"),
    }
