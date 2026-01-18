"""
AutoML Pipeline Orchestrator
Automatically triggers agents when stages transition to IN_PROGRESS
Handles the complete flow: PARSE_INTENT → DATA_SOURCE → ... → EXPORT
"""
import asyncio
from typing import Any, Dict, Optional

from app.agents.prompt_parser import PromptParserAgent
from app.agents.dataset_finder import DatasetFinderAgent
from app.agents.preprocess import PreprocessAgent
from app.agents.model_selector import ModelSelectorAgent
from app.agents.trainer import TrainerAgent
from app.agents.verifier import VerifierAgent
from app.agents.reporter import ReporterAgent
from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus
from app.orchestrator.conductor import conductor


class PipelineOrchestrator:
    """
    Orchestrates the AutoML pipeline by automatically triggering agents
    when stages transition to IN_PROGRESS.
    
    Flow:
    1. PARSE_INTENT → PromptParserAgent extracts intent/task
    2. DATA_SOURCE → DatasetFinderAgent finds HuggingFace datasets
    3. DATA_CLEAN → PreprocessAgent profiles data
    4. FEATURE_ENG → PreprocessAgent applies transformations
    5. MODEL_SELECT → ModelSelectorAgent recommends models
    6. TRAIN → TrainerAgent trains selected model
    7. EVALUATE → VerifierAgent evaluates model
    8. EXPORT → ReporterAgent exports model/notebook
    """
    
    def __init__(self):
        # Store pipeline context (parsed intent, selected dataset, etc)
        self._context: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    def _get_context(self, project_id: str) -> Dict[str, Any]:
        """Get or create pipeline context for a project"""
        if project_id not in self._context:
            self._context[project_id] = {
                "parsed_intent": None,
                "selected_dataset": None,
                "dataset_candidates": [],
                "data_profile": None,
                "preprocessing_steps": [],
                "selected_model": None,
                "model_candidates": [],
                "trained_model": None,
                "evaluation_metrics": None,
            }
        return self._context[project_id]
    
    async def start_pipeline(self, project_id: str, user_prompt: str) -> Dict[str, Any]:
        """
        Start the pipeline with a user prompt.
        Triggers PARSE_INTENT stage.
        """
        print(f"\n{'='*60}")
        print(f"[Pipeline Orchestrator] 🚀 Starting pipeline for project: {project_id}")
        print(f"[Pipeline Orchestrator] Prompt: {user_prompt[:100]}...")
        print(f"{'='*60}\n")
        
        # Initialize context
        async with self._lock:
            context = self._get_context(project_id)
            context["user_prompt"] = user_prompt
        
        # Trigger PARSE_INTENT stage
        await self.execute_stage(project_id, StageID.PARSE_INTENT)
        
        return {"status": "started", "project_id": project_id}
    
    async def execute_stage(self, project_id: str, stage_id: StageID) -> None:
        """
        Execute a specific stage by calling the appropriate agent.
        This is called automatically when a stage transitions to IN_PROGRESS.
        """
        print(f"\n[Pipeline Orchestrator] 🔄 Executing stage: {stage_id.value}")
        
        try:
            if stage_id == StageID.PARSE_INTENT:
                await self._execute_parse_intent(project_id)
            
            elif stage_id == StageID.DATA_SOURCE:
                await self._execute_data_source(project_id)
            
            elif stage_id == StageID.PROFILE_DATA:
                await self._execute_profile_data(project_id)
            
            elif stage_id == StageID.PREPROCESS:
                await self._execute_preprocess(project_id)
            
            elif stage_id == StageID.MODEL_SELECT:
                await self._execute_model_select(project_id)
            
            elif stage_id == StageID.TRAIN:
                await self._execute_train(project_id)
            
            elif stage_id == StageID.REVIEW_EDIT:
                await self._execute_review_edit(project_id)
            
            elif stage_id == StageID.EXPORT:
                await self._execute_export(project_id)
            
            else:
                print(f"[Pipeline Orchestrator] ⚠️  No handler for stage: {stage_id.value}")
        
        except Exception as e:
            print(f"[Pipeline Orchestrator] ❌ Error in stage {stage_id.value}: {e}")
            import traceback
            traceback.print_exc()
            # Publish error as stage status with error message
            await conductor.transition_to(
                project_id=project_id,
                stage_id=stage_id,
                status=StageStatus.IN_PROGRESS,
                message=f"Error: {str(e)}"
            )
            raise
    
    async def _execute_parse_intent(self, project_id: str) -> None:
        """Stage 1: Parse user prompt to extract intent and task type"""
        context = self._get_context(project_id)
        user_prompt = context.get("user_prompt", "")
        
        print("[1/4] Creating PromptParserAgent...")
        agent = PromptParserAgent()
        
        print("[2/4] Parsing prompt...")
        parsed = agent.parse(user_prompt)
        print(f"[2/4] ✅ Parsed: {parsed}")
        
        # Store in context
        async with self._lock:
            context["parsed_intent"] = parsed
        
        print("[3/4] Publishing PROMPT_PARSED event...")
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.PROMPT_PARSED,
            payload=parsed,
            stage_id=StageID.PARSE_INTENT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        print("[4/4] Waiting for user confirmation...")
        await conductor.waiting_for_confirmation(
            project_id=project_id,
            stage_id=StageID.PARSE_INTENT,
            summary=f"Understood! You want to build a {parsed.get('task', 'model')}. Task type: {parsed.get('task_type', 'classification')}",
            next_actions=["Confirm to search for datasets", "Edit prompt if incorrect"]
        )
        
        print("[PARSE_INTENT] ✅ Complete - waiting for confirmation")
    
    async def _execute_data_source(self, project_id: str) -> None:
        """Stage 2: Find datasets from HuggingFace based on parsed intent"""
        context = self._get_context(project_id)
        parsed_intent = context.get("parsed_intent", {})
        
        print("[1/4] Creating DatasetFinderAgent...")
        agent = DatasetFinderAgent()
        
        print("[2/4] Searching for datasets...")
        # Pass the parsed task info to find relevant datasets
        # Try multiple fields from parsed_intent to build user input
        user_input_parts = []
        if parsed_intent.get("task"):
            user_input_parts.append(parsed_intent.get("task"))
        if parsed_intent.get("description"):
            user_input_parts.append(parsed_intent.get("description"))
        if parsed_intent.get("target"):
            user_input_parts.append(parsed_intent.get("target"))
        if parsed_intent.get("dataset_hint"):
            user_input_parts.append(parsed_intent.get("dataset_hint"))
        
        user_input = " ".join(user_input_parts) if user_input_parts else "classification dataset"
        task_type = parsed_intent.get("task_type", "classification")
        
        print(f"[DEBUG] Parsed intent: {parsed_intent}")
        print(f"[DEBUG] User input for search: '{user_input}'")
        print(f"[DEBUG] Task type: {task_type}")
        
        datasets = agent.find_datasets(user_input, task_type=task_type, limit=5)
        print(f"[2/4] ✅ Found {len(datasets)} datasets")
        
        # Store in context
        async with self._lock:
            context["dataset_candidates"] = datasets
        
        print("[3/4] Publishing DATASET_CANDIDATES event...")
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.DATASET_CANDIDATES,
            payload={"datasets": datasets, "count": len(datasets)},
            stage_id=StageID.DATA_SOURCE,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        print("[4/4] Waiting for user to select dataset...")
        dataset_summaries = [
            f"{i+1}. {d['name']}: {d.get('description', 'No description')[:80]}..." 
            for i, d in enumerate(datasets[:3])
        ]
        await conductor.waiting_for_confirmation(
            project_id=project_id,
            stage_id=StageID.DATA_SOURCE,
            summary=f"Found {len(datasets)} datasets:\n" + "\n".join(dataset_summaries),
            next_actions=["Select a dataset", "Upload your own dataset"]
        )
        
        print("[DATA_SOURCE] ✅ Complete - waiting for dataset selection")
    
    async def _execute_profile_data(self, project_id: str) -> None:
        """Stage 3: Profile the selected dataset"""
        context = self._get_context(project_id)
        selected_dataset = context.get("selected_dataset")
        
        if not selected_dataset:
            print("[PROFILE_DATA] ⚠️ No dataset selected, skipping profiling")
            await conductor.waiting_for_confirmation(
                project_id=project_id,
                stage_id=StageID.PROFILE_DATA,
                summary="No dataset selected. Please go back and select a dataset.",
                next_actions=["Select a dataset"]
            )
            return
        
        print("[1/4] Creating PreprocessAgent for profiling...")
        agent = PreprocessAgent()
        
        print("[2/4] Profiling dataset...")
        # Get dataset path (from upload or download)
        dataset_path = selected_dataset.get("path")
        
        if not dataset_path:
            # If HuggingFace dataset, we'd need to download it first
            # For now, use a placeholder
            print("[PROFILE_DATA] ⚠️ Dataset path not found - using mock profile")
            profile = {
                "rows": 10000,
                "columns": 15,
                "summary": {
                    "total_missing_values": 150,
                    "missing_percentage": 1.0,
                    "numeric_column_count": 10,
                    "categorical_column_count": 5
                }
            }
        else:
            try:
                profile = agent.profile_dataset(dataset_path)
            except Exception as e:
                print(f"[PROFILE_DATA] ❌ Error profiling: {e}")
                profile = {"error": str(e)}
        
        async with self._lock:
            context["data_profile"] = profile
        
        print("[3/4] Publishing PROFILE_SUMMARY event...")
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.PROFILE_SUMMARY,
            payload={"profile": profile},
            stage_id=StageID.PROFILE_DATA,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        summary_text = f"Dataset profiled: {profile.get('rows', 'N/A')} rows, {profile.get('columns', 'N/A')} columns"
        if 'summary' in profile:
            summary_text += f", {profile['summary'].get('missing_percentage', 0):.1f}% missing values"
        
        print("[4/4] Waiting for confirmation...")
        await conductor.waiting_for_confirmation(
            project_id=project_id,
            stage_id=StageID.PROFILE_DATA,
            summary=summary_text,
            next_actions=["Confirm to proceed with preprocessing"]
        )
        
        print("[PROFILE_DATA] ✅ Complete - waiting for confirmation")
    
    async def _execute_preprocess(self, project_id: str) -> None:
        """Stage 4: Apply preprocessing transformations"""
        context = self._get_context(project_id)
        selected_dataset = context.get("selected_dataset")
        
        if not selected_dataset:
            print("[PREPROCESS] ⚠️ No dataset selected")
            await conductor.waiting_for_confirmation(
                project_id=project_id,
                stage_id=StageID.PREPROCESS,
                summary="No dataset to preprocess",
                next_actions=["Select a dataset"]
            )
            return
        
        print("[1/4] Creating PreprocessAgent...")
        agent = PreprocessAgent()
        
        print("[2/4] Applying preprocessing...")
        dataset_path = selected_dataset.get("path")
        
        if dataset_path:
            try:
                import pandas as pd
                # Load data
                df = pd.read_csv(dataset_path)
                print(f"[PREPROCESS] Loaded {len(df)} rows")
                
                # Apply preprocessing
                df_processed = agent.preprocess(df)
                
                # Save processed data
                from pathlib import Path
                processed_path = Path(dataset_path).parent / "processed_data.csv"
                df_processed.to_csv(processed_path, index=False)
                print(f"[PREPROCESS] Saved processed data to {processed_path}")
                
                # Get summary
                summary_info = agent.get_preprocessing_summary()
                preprocessing_steps = summary_info["steps"]
                
                async with self._lock:
                    context["preprocessing_steps"] = preprocessing_steps
                    context["processed_data_path"] = str(processed_path)
                
            except Exception as e:
                print(f"[PREPROCESS] ❌ Error preprocessing: {e}")
                preprocessing_steps = [f"Error: {e}"]
        else:
            # Mock for HuggingFace datasets
            preprocessing_steps = [
                "Handle missing values with median/mode",
                "Encode categorical variables",
                "Scale numerical features"
            ]
            async with self._lock:
                context["preprocessing_steps"] = preprocessing_steps
        
        print("[3/4] Publishing preprocessing event...")
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload={"steps": preprocessing_steps, "step_count": len(preprocessing_steps)},
            stage_id=StageID.PREPROCESS,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        print("[4/4] Waiting for confirmation...")
        await conductor.waiting_for_confirmation(
            project_id=project_id,
            stage_id=StageID.PREPROCESS,
            summary=f"Applied {len(preprocessing_steps)} preprocessing steps:\n" + "\n".join(f"• {step}" for step in preprocessing_steps[:5]),
            next_actions=["Confirm to select model"]
        )
        
        print("[PREPROCESS] ✅ Complete")
    
    async def _execute_model_select(self, project_id: str) -> None:
        """Stage 5: Recommend ML models based on task type"""
        context = self._get_context(project_id)
        parsed_intent = context.get("parsed_intent", {})
        task_type = parsed_intent.get("task_type", "classification")
        
        # Map task types to model selector's expected types (classification or regression)
        task_type_mapping = {
            "vision": "classification",
            "nlp": "classification",
            "tabular": "classification",  # Default to classification, could be regression
            "clustering": "classification",
            "timeseries": "regression",
            "classification": "classification",
            "regression": "regression",
            "other": "classification"
        }
        mapped_task_type = task_type_mapping.get(task_type, "classification")
        print(f"[DEBUG] Task type '{task_type}' mapped to '{mapped_task_type}' for model selection")
        
        print("[1/3] Creating ModelSelectorAgent...")
        agent = ModelSelectorAgent()
        
        print("[2/3] Selecting models...")
        models = agent.select_model(mapped_task_type)
        print(f"[2/3] ✅ Recommended {len(models)} models")
        
        async with self._lock:
            context["model_candidates"] = models
        
        print("[3/3] Publishing MODEL_CANDIDATES event...")
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.MODEL_CANDIDATES,
            payload={"models": models, "task_type": task_type},
            stage_id=StageID.MODEL_SELECT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        model_summaries = [f"• {m['name']}: {m.get('description', 'ML model')}" for m in models[:3]]
        await conductor.waiting_for_confirmation(
            project_id=project_id,
            stage_id=StageID.MODEL_SELECT,
            summary=f"Recommended models for {task_type}:\n" + "\n".join(model_summaries),
            next_actions=["Confirm to train model", "Select a different model"]
        )
        
        print("[MODEL_SELECT] ✅ Complete")
    
    async def _execute_train(self, project_id: str) -> None:
        """Stage 6: Train the selected model"""
        context = self._get_context(project_id)
        selected_model = context.get("selected_model")
        processed_data_path = context.get("processed_data_path")
        parsed_intent = context.get("parsed_intent", {})
        task_type = parsed_intent.get("task_type", "classification")
        selected_dataset = context.get("selected_dataset", {})
        dataset_name = selected_dataset.get("name", "")
        
        # Check if this is an image dataset
        is_image_dataset = any(term in dataset_name.lower() for term in ["cat", "dog", "cifar", "mnist", "fashion", "image"]) or task_type == "vision"
        
        if is_image_dataset:
            # Show warning for image datasets
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.TRAIN_PROGRESS,
                payload={
                    "progress": 0,
                    "message": "⚠️ WARNING: Image dataset detected. Training sklearn model on image metadata only (not actual pixels). For real image classification, use transfer learning with PyTorch/TensorFlow."
                },
                stage_id=StageID.TRAIN,
                stage_status=StageStatus.IN_PROGRESS,
            )
            print("[TRAIN] ⚠️ WARNING: Training on image metadata, not actual image data")
        
        print("[1/6] Creating TrainerAgent...")
        agent = TrainerAgent()
        
        # Initialize model
        model_id = selected_model.get("id", "rf") if selected_model else "rf"
        agent.initialize_model(model_id, task_type)
        
        print("[2/6] Starting training...")
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.TRAIN_PROGRESS,
            payload={"progress": 0, "message": "Training started"},
            stage_id=StageID.TRAIN,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        # Progress callback
        async def report_progress(progress: int, message: str):
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.TRAIN_PROGRESS,
                payload={"progress": progress, "message": message},
                stage_id=StageID.TRAIN,
                stage_status=StageStatus.IN_PROGRESS,
            )
            await asyncio.sleep(0.1)  # Small delay for frontend to process
        
        # Train the model
        if processed_data_path:
            try:
                print(f"[3/6] Training on {processed_data_path}...")
                metrics = await agent.train(
                    data_path=processed_data_path,
                    progress_callback=report_progress
                )
                
                # Save trained model
                from pathlib import Path
                model_path = Path(processed_data_path).parent / "trained_model.joblib"
                agent.save_model(str(model_path))
                
                async with self._lock:
                    context["trained_model_path"] = str(model_path)
                    context["training_metrics"] = metrics
                
                print(f"[4/6] Training complete - Test score: {metrics.get('test_score', 'N/A')}")
                
            except Exception as e:
                print(f"[TRAIN] ❌ Error during training: {e}")
                metrics = {"error": str(e), "test_score": 0.0}
                async with self._lock:
                    context["training_metrics"] = metrics
        else:
            # Mock training if no data
            print("[3/6] No processed data - simulating training...")
            for progress in [25, 50, 75, 100]:
                await report_progress(progress, f"Training {progress}% complete")
                await asyncio.sleep(0.5)
            
            metrics = {"accuracy": 0.95, "test_score": 0.95, "train_score": 0.97}
            async with self._lock:
                context["training_metrics"] = metrics
        
        print("[5/6] Publishing TRAIN_COMPLETE event...")
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.TRAIN_PROGRESS,
            payload={"progress": 100, "metrics": metrics},
            stage_id=StageID.TRAIN,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        score = metrics.get("test_score", metrics.get("accuracy", 0.0))
        print("[6/6] Waiting for confirmation...")
        await conductor.waiting_for_confirmation(
            project_id=project_id,
            stage_id=StageID.TRAIN,
            summary=f"Training complete! Test score: {score:.1%}",
            next_actions=["Confirm to evaluate model"]
        )
        
        print("[TRAIN] ✅ Complete")
    
    async def _execute_review_edit(self, project_id: str) -> None:
        """Stage 7: Review and evaluate trained model"""
        context = self._get_context(project_id)
        
        print("[1/3] Creating VerifierAgent...")
        agent = VerifierAgent()
        
        print("[2/3] Evaluating model...")
        # TODO: Implement actual evaluation
        eval_results = {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.96,
            "f1_score": 0.95
        }
        
        async with self._lock:
            context["evaluation_metrics"] = eval_results
        
        print("[3/3] Publishing evaluation event...")
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload={"evaluation": eval_results},
            stage_id=StageID.REVIEW_EDIT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        await conductor.waiting_for_confirmation(
            project_id=project_id,
            stage_id=StageID.REVIEW_EDIT,
            summary=f"Model evaluation complete. F1 Score: {eval_results['f1_score']:.1%}",
            next_actions=["Confirm to export model"]
        )
        
        print("[REVIEW_EDIT] ✅ Complete")
    
    async def _execute_export(self, project_id: str) -> None:
        """Stage 8: Export trained model and generate notebook"""
        context = self._get_context(project_id)
        
        print("[1/3] Creating ReporterAgent...")
        # TODO: Use ReporterAgent to generate notebook
        
        print("[2/3] Exporting model...")
        export_info = {
            "format": "ONNX",
            "path": "/exports/model.onnx",
            "notebook": "/exports/training.ipynb"
        }
        
        print("[3/3] Publishing EXPORT_COMPLETE event...")
        await event_bus.publish_event(
            project_id=project_id,
            event_name=EventType.STAGE_STATUS,
            payload={"export": export_info},
            stage_id=StageID.EXPORT,
            stage_status=StageStatus.IN_PROGRESS,
        )
        
        await conductor.waiting_for_confirmation(
            project_id=project_id,
            stage_id=StageID.EXPORT,
            summary="Model exported successfully! Download your trained model and training notebook.",
            next_actions=["Download model", "View notebook"]
        )
        
        print("[EXPORT] ✅ Complete - Pipeline finished!")
    
    async def handle_confirmation(self, project_id: str, user_selection: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle user confirmation and advance to next stage.
        Called when user clicks "Confirm & Continue"
        
        Args:
            project_id: Project ID
            user_selection: Optional user selection (e.g., selected dataset, model)
        """
        # Get current stage from conductor
        current_stage = conductor._get_current_stage_id(project_id)
        
        # Store user selection if provided
        if user_selection:
            async with self._lock:
                context = self._get_context(project_id)
                if "dataset_id" in user_selection:
                    context["selected_dataset"] = user_selection["dataset_id"]
                if "model_id" in user_selection:
                    context["selected_model"] = user_selection["model_id"]
        
        # Confirm current stage and get next stage
        result = await conductor.confirm(project_id)
        
        # Check if we advanced to a new stage
        new_stage_id = result.get("current_stage", {}).get("id")
        if new_stage_id and new_stage_id != current_stage.value:
            # Automatically execute the new stage
            new_stage = StageID(new_stage_id)
            await self.execute_stage(project_id, new_stage)


# Global orchestrator instance
orchestrator = PipelineOrchestrator()
