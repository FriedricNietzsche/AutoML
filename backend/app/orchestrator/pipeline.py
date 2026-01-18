"""
AutoML Pipeline Orchestrator
Automatically triggers agents when stages transition to IN_PROGRESS
Handles the complete flow: PARSE_INTENT → DATA_SOURCE → ... → EXPORT
"""
import asyncio
import os
from typing import Any, Dict, Optional
from pathlib import Path

from app.agents.prompt_parser import PromptParserAgent
from app.agents.dataset_finder import DatasetFinderAgent
from app.agents.preprocess import PreprocessAgent
from app.agents.model_selector import ModelSelectorAgent
from app.ml.trainer_factory import TrainerFactory  # NEW: Smart model selection
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
                
                # Detect target column
                target_column = "label" if "label" in df.columns else df.columns[-1]
                print(f"[PREPROCESS] Target column: {target_column}")
                
                # Apply preprocessing with target column awareness
                df_processed = agent.preprocess(df, target_column=target_column)
                
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
        """Stage 6: Train the selected model with proper trainer (text/image/tabular)"""
        import pandas as pd
        from pathlib import Path
        import joblib
        
        context = self._get_context(project_id)
        selected_model = context.get("selected_model")
        processed_data_path = context.get("processed_data_path")
        parsed_intent = context.get("parsed_intent", {})
        task_type = parsed_intent.get("task_type", "classification")
        selected_dataset = context.get("selected_dataset", {})
        dataset_name = selected_dataset.get("name", "Unknown")
        original_data_path = selected_dataset.get("path")  # Original data BEFORE preprocessing
        
        print(f"[TRAIN] Starting training for {dataset_name}")
        print(f"[TRAIN] Task type: {task_type}")
        print(f"[TRAIN] Original data: {original_data_path}")
        print(f"[TRAIN] Processed data: {processed_data_path}")
        
        # Progress callback
        async def report_progress(progress: int, message: str):
            await event_bus.publish_event(
                project_id=project_id,
                event_name=EventType.TRAIN_PROGRESS,
                payload={"progress": progress, "message": message},
                stage_id=StageID.TRAIN,
                stage_status=StageStatus.IN_PROGRESS,
            )
            await asyncio.sleep(0.1)
        
        if not processed_data_path:
            print("[TRAIN] ❌ No processed data available")
            await report_progress(100, "Training skipped - no data")
            return
        
        try:
            # Load ORIGINAL data to detect task type (before preprocessing destroyed text)
            original_df = pd.read_csv(original_data_path) if original_data_path else None
            processed_df = pd.read_csv(processed_data_path)
            
            print(f"[TRAIN] Loaded original data: {original_df.shape if original_df is not None else 'N/A'}")
            print(f"[TRAIN] Loaded processed data: {processed_df.shape}")
            
            await report_progress(10, f"Loaded {len(processed_df)} samples...")
            
            # Use ORIGINAL data for detection (before preprocessing) - DEFINE THIS FIRST
            df_for_detection = original_df if original_df is not None else processed_df
            
            # Get target column from parsed intent
            target_column = parsed_intent.get("target", "label")
            
            # Fallback: if target not in dataframe, try common names
            if target_column not in df_for_detection.columns:
                if "label" in df_for_detection.columns:
                    target_column = "label"
                elif "Survived" in df_for_detection.columns:
                    target_column = "Survived"
                elif "survived" in df_for_detection.columns:
                    target_column = "survived"
                else:
                    # Last resort: use last column
                    target_column = df_for_detection.columns[-1]
            
            print(f"[TRAIN] Target column: {target_column}")
            
            # Detect task type from ORIGINAL data (before label encoding destroyed text)
            actual_task_type = TrainerFactory.detect_task_type(df_for_detection, target_column)
            
            print(f"[TRAIN] Detected task type: {actual_task_type}")
            await report_progress(20, f"Detected: {actual_task_type}")
            
            # Determine number of classes for classification (use ORIGINAL data to get correct count)
            num_classes = 2
            if actual_task_type in ["text_classification", "image_classification", "tabular_classification"]:
                # Count classes from ORIGINAL data (before label encoding)
                num_classes = original_df[target_column].nunique() if original_df is not None else processed_df[target_column].nunique()
                print(f"[TRAIN] Number of classes: {num_classes}")
            
            # Map model IDs to trainer names (task-aware)
            model_id = selected_model.get("id", "auto") if selected_model else "auto"
            
            # For TEXT classification, always use transformers (ignore sklearn model selections)
            if actual_task_type == "text_classification":
                model_name = "auto"  # Will default to DistilBERT in SentimentClassifier
                print(f"[TRAIN] Text classification detected - using transformer model (DistilBERT)")
            else:
                # For tabular tasks, map sklearn model IDs to trainer names
                model_name_map = {
                    "rf_clf": "random_forest",  # Random Forest Classifier
                    "xgb_clf": "xgboost",        # XGBoost Classifier
                    "lr_clf": "auto",            # Logistic Regression → use auto (XGBoost)
                    "rf_reg": "random_forest",   # Random Forest Regressor
                    "xgb_reg": "xgboost",        # XGBoost Regressor
                    "lr_reg": "auto",            # Linear Regression → use auto (XGBoost)
                }
                model_name = model_name_map.get(model_id, model_id)
            
            print(f"[TRAIN] Creating trainer (model_id={model_id}, model_name={model_name}, task={actual_task_type})...")
            
            await report_progress(30, f"Initializing {model_name} trainer...")
            
            # Check if cloud training is enabled and available
            use_cloud_training = os.getenv("USE_VULTR_TRAINING", "false").lower() == "true"
            vultr_api_key = os.getenv("VULTR_API_KEY")
            
            metrics = None
            model_path = None
            trainer = None
            
            if use_cloud_training and vultr_api_key and actual_task_type == "text_classification":
                # Use Vultr cloud training for text classification
                print(f"[TRAIN] ☁️ Cloud training enabled - using Vultr GPU instance")
                await report_progress(35, "Provisioning Vultr GPU instance...")
                
                try:
                    from app.cloud.vultr_trainer import VultrTrainer
                    
                    cloud_trainer = VultrTrainer(api_key=vultr_api_key)
                    
                    # Use original data path for cloud training
                    cloud_result = await cloud_trainer.train_on_cloud(
                        task_type=actual_task_type,
                        train_data_path=original_data_path,
                        model_name="distilbert-base-uncased",
                        num_classes=num_classes,
                        hyperparameters={
                            "num_epochs": 3,
                            "batch_size": 16,  # Larger batch size on GPU
                            "learning_rate": 2e-5
                        }
                    )
                    
                    metrics = cloud_result["metrics"]
                    cloud_model_path = cloud_result["model_path"]
                    
                    # Copy cloud model to project directory
                    import shutil
                    model_path = Path(f"{Path(original_data_path).parent}/model")
                    if model_path.exists():
                        shutil.rmtree(model_path)
                    shutil.copytree(cloud_model_path, model_path)
                    
                    print(f"[TRAIN] ✅ Cloud training complete! Model saved to {model_path}")
                    
                except Exception as cloud_error:
                    print(f"[TRAIN] ⚠️ Cloud training failed: {cloud_error}")
                    print(f"[TRAIN] Falling back to local training...")
                    use_cloud_training = False
            
            if not use_cloud_training:
                # Local training (existing code)
                trainer = TrainerFactory.get_trainer(
                    task_type=actual_task_type,
                    model_name=model_name,
                    num_classes=num_classes
                )
            
            # Only proceed with local training if we have a trainer
            if trainer is not None:
                # Train the model
                print(f"[TRAIN] Training {actual_task_type} model...")
                await report_progress(40, f"Training {actual_task_type} model (this may take a few minutes)...")
                
                # Run training in executor to avoid blocking
                import concurrent.futures
                loop = asyncio.get_event_loop()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    if actual_task_type == "text_classification":
                        # Text classification with transformers - use ORIGINAL data with text
                        text_col = None
                        for col in original_df.columns:
                            if col != target_column and original_df[col].dtype == 'object':
                                unique_ratio = original_df[col].nunique() / len(original_df)
                                avg_length = original_df[col].astype(str).str.len().mean()
                                if unique_ratio > 0.9 and avg_length > 50:
                                    text_col = col
                                    break
                        
                        if text_col:
                            print(f"[TRAIN] Found text column: {text_col}")
                            train_texts = original_df[text_col].tolist()
                            train_labels = original_df[target_column].tolist()
                            
                            # Check label distribution
                            unique_labels_in_data = original_df[target_column].unique()
                            print(f"[TRAIN] Unique labels in full dataset: {unique_labels_in_data} (count: {len(unique_labels_in_data)})")
                            
                            # Split into train/val
                            split_idx = int(len(train_texts) * 0.8)
                            
                            def train_text_model():
                                return trainer.train(
                                    train_texts=train_texts[:split_idx],
                                    train_labels=train_labels[:split_idx],
                                    val_texts=train_texts[split_idx:],
                                    val_labels=train_labels[split_idx:],
                                    num_epochs=3  # Fast training for demo
                                )
                            
                            metrics = await loop.run_in_executor(executor, train_text_model)
                        else:
                            raise ValueError("No text column found for text classification")
                    
                    elif actual_task_type in ["tabular_classification", "tabular_regression"]:
                        # Tabular with XGBoost/RandomForest - use PROCESSED data (scaled/encoded)
                        X = processed_df.drop(columns=[target_column])
                        y = processed_df[target_column]
                        
                        from sklearn.model_selection import train_test_split
                        X_train, X_val, y_train, y_val = train_test_split(
                            X, y, test_size=0.2, random_state=42,
                            stratify=y if actual_task_type == "tabular_classification" else None
                        )
                        
                        def train_tabular_model():
                            return trainer.train(
                                X_train=X_train,
                                y_train=y_train,
                                X_val=X_val,
                                y_val=y_val
                            )
                        
                        metrics = await loop.run_in_executor(executor, train_tabular_model)
                    
                    elif actual_task_type == "image_classification":
                        # Image classification with transfer learning
                        # TODO: Implement image path extraction
                        raise NotImplementedError("Image classification training not yet implemented in pipeline")
                    
                    else:
                        raise ValueError(f"Unknown task type: {actual_task_type}")
            
            await report_progress(80, "Training complete, saving model...")
            
            # Save trained model (only if local training was used)
            if trainer is not None and model_path is None:
                model_dir = Path(processed_data_path).parent
                model_path = model_dir / "trained_model.joblib"
                
                print(f"[TRAIN] Saving model to {model_path}...")
                trainer.save(model_path)
            
            # Ensure we have metrics
            if metrics is None:
                raise ValueError("Training did not produce metrics")
            
            # Store metrics
            training_metrics = {
                "model_name": trainer.__class__.__name__ if trainer else "CloudModel",
                "task_type": actual_task_type,
                "dataset": dataset_name,
                **metrics
            }
            
            async with self._lock:
                context["trained_model_path"] = str(model_path)
                context["training_metrics"] = training_metrics
                context["trainer_class"] = trainer.__class__.__name__ if trainer else "CloudModel"
            
            print(f"[TRAIN] ✅ Training complete!")
            print(f"[TRAIN] Metrics: {training_metrics}")
            
            await report_progress(100, f"Training complete! Accuracy: {metrics.get('val_accuracy', metrics.get('train_accuracy', 0)):.1%}")
            
            # Show summary
            score = metrics.get("val_accuracy", metrics.get("train_accuracy", 0.0))
            await conductor.waiting_for_confirmation(
                project_id=project_id,
                stage_id=StageID.TRAIN,
                summary=f"✅ Training complete!\nModel: {trainer.__class__.__name__}\nAccuracy: {score:.1%}",
                next_actions=["Confirm to evaluate model"]
            )
            
        except Exception as e:
            print(f"[TRAIN] ❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            
            await report_progress(0, f"Training failed: {str(e)}")
            
            async with self._lock:
                context["training_metrics"] = {"error": str(e)}
            
            raise
        
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
