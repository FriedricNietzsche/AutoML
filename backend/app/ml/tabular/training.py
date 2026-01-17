"""
Training Runner Module (Task 5.1 - Main Implementation)

Trains XGBoost models with REAL streaming progress using callbacks.
Emits all required Stage 3 events for live training visualization.

Events emitted (in order):
  1. TRAIN_RUN_STARTED
  2. LOG_LINE (multiple)
  3. TRAIN_PROGRESS (multiple, real-time)
  4. METRIC_SCALAR (multiple, real-time from XGBoost)
  5. RESOURCE_STATS (periodic)
  6. CONFUSION_MATRIX_READY / RESIDUALS_PLOT_READY
  7. FEATURE_IMPORTANCE_READY
  8. BEST_MODEL_UPDATED
  9. LEADERBOARD_UPDATED (if comparing models)
  10. TRAIN_RUN_FINISHED
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

from .metrics import compute_all_metrics
from .visualization import (
    generate_confusion_matrix_plot,
    generate_residuals_plot,
    generate_feature_importance_plot,
    generate_training_curves,
)
from .model_registry import ModelRegistry, create_metadata_from_training
from .leaderboard import LeaderboardManager
from .report_generator import ReportGenerator


# ============================================================================
# Pydantic Event Models (Contract-compliant)
# ============================================================================

class TrainRunStarted(BaseModel):
    """TRAIN_RUN_STARTED event payload."""
    run_id: str
    model_id: str
    metric_primary: str
    config: Dict[str, Any]


class TrainProgress(BaseModel):
    """TRAIN_PROGRESS event payload."""
    run_id: str
    epoch: int
    epochs: int
    step: int
    steps: int
    eta_s: Optional[float] = None
    phase: str  # init|fit|eval|finalize


class MetricScalar(BaseModel):
    """METRIC_SCALAR event payload."""
    run_id: str
    name: str
    split: str  # train|val|test
    step: int
    value: float


class ResourceStats(BaseModel):
    """RESOURCE_STATS event payload."""
    run_id: str
    cpu_pct: float
    ram_mb: float
    gpu_pct: float
    vram_mb: float
    step_per_sec: float


class LogLine(BaseModel):
    """LOG_LINE event payload."""
    run_id: str
    level: str  # info|warning|error
    text: str


class BestModelUpdated(BaseModel):
    """BEST_MODEL_UPDATED event payload."""
    run_id: str
    model_id: str
    metric: Dict[str, Any]  # {name, split, value}


class TrainRunFinished(BaseModel):
    """TRAIN_RUN_FINISHED event payload."""
    run_id: str
    status: str  # success|failed|cancelled
    final_metrics: Dict[str, Any]


# ============================================================================
# TrainingRunner - Main Training Orchestrator
# ============================================================================

class TrainingRunner:
    """
    Trains XGBoost models with real streaming progress.
    
    Uses XGBoost's built-in callbacks to emit REAL progress updates
    as each boosting iteration completes.
    
    Usage:
        # Create event emitter
        def emit_fn(event_name, payload):
            print(f"{event_name}: {payload}")
        
        # Create runner
        runner = TrainingRunner(emit_event=emit_fn)
        
        # Train model
        result = await runner.run_training(
            df=dataframe,
            target_column="churn",
            task_type="classification",
            model_id="xgboost_classifier",
            preprocessor=column_transformer,
            output_dir="assets/project_123/training"
        )
    """
    
    def __init__(
        self,
        emit_event: Callable[[str, Dict], None],
        project_id: str = "default_project",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize training runner.
        
        Args:
            emit_event: Callback function to emit events (event_name, payload_dict)
            project_id: Project identifier for model registry
            test_size: Fraction of data for test set
            val_size: Fraction of remaining data for validation
            random_state: Random seed for reproducibility
        """
        self.emit = emit_event
        self.project_id = project_id
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Task 5.3: Initialize registry, leaderboard, and report generator
        self.registry = ModelRegistry()
        self.leaderboard = LeaderboardManager()
        self.report_gen = ReportGenerator()
        
        # Training state
        self.current_run_id = None
        self.start_time = None
        self.iteration_times = []
    
    async def run_training(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
        model_id: str,
        preprocessor: Any,
        output_dir: str,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run complete training workflow with real streaming.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            task_type: 'classification' or 'regression'
            model_id: Model identifier (e.g., 'xgboost_classifier')
            preprocessor: Fitted sklearn ColumnTransformer
            output_dir: Directory to save artifacts
            n_estimators: Number of boosting rounds (each round = 1 progress update)
            max_depth: Tree depth
            learning_rate: Learning rate
            
        Returns:
            Dict with run_id, model_path, metrics, artifact_paths
        """
        # Generate unique run ID
        self.current_run_id = f"run_{uuid.uuid4().hex[:8]}"
        self.start_time = time.time()
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Phase 1: Initialize
            await self._phase_init(
                df, target_column, task_type, model_id, 
                n_estimators, max_depth, learning_rate
            )
            
            # Phase 2: Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(
                df, target_column, task_type
            )
            
            # Phase 3: Fit (with REAL streaming callbacks)
            model, history = await self._phase_fit(
                X_train, y_train, X_val, y_val,
                task_type, preprocessor,
                n_estimators, max_depth, learning_rate
            )
            
            # Phase 4: Evaluate
            metrics_dict = await self._phase_evaluate(
                model, X_test, y_test, task_type
            )
            
            # Phase 5: Generate artifacts
            artifact_paths = await self._phase_finalize(
                model, X_test, y_test, metrics_dict, task_type,
                output_path, history
            )
            
            # Calculate training duration
            training_duration = time.time() - self.start_time
            
            # Task 5.3: Save model with metadata to registry
            metadata = create_metadata_from_training(
                run_id=self.current_run_id,
                project_id=self.project_id,
                model_id=model_id,
                model_family="XGBoost",
                task_type=task_type,
                metrics=metrics_dict["metrics_dict"],
                primary_metric_name=metrics_dict["primary_metric_name"],
                hyperparameters={
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate
                },
                training_config={
                    "test_size": self.test_size,
                    "val_size": self.val_size,
                    "random_state": self.random_state,
                    "target_column": target_column
                },
                artifact_paths=artifact_paths,
                n_train_samples=len(X_train),
                n_val_samples=len(X_val),
                n_test_samples=len(X_test),
                n_features=X_train.shape[1],
                training_duration_seconds=training_duration
            )
            
            # Save to registry
            self.registry.save_model_with_metadata(
                self.current_run_id,
                self.project_id,
                model,
                metadata
            )
            
            # Task 5.3: Generate comprehensive report
            report_path = self.report_gen.generate_report_json(
                self.current_run_id,
                self.project_id,
                metadata
            )
            
            # Task 5.3: Update leaderboard
            self.leaderboard.add_run(self.project_id, metadata)
            
            # Task 5.3: Emit LEADERBOARD_UPDATED event
            leaderboard_entries = self.leaderboard.get_leaderboard(
                self.project_id,
                top_n=10
            )
            self.leaderboard.emit_leaderboard_event(
                self.emit,
                leaderboard_entries,
                metadata.primary_metric_name
            )
            
            # Task 5.3: Check if this is the best model
            best_run = self.leaderboard.get_best_run(self.project_id)
            if best_run and best_run.run_id == self.current_run_id:
                self.leaderboard.emit_best_model_event(
                    self.emit,
                    self.current_run_id,
                    model_id,
                    metadata.primary_metric_name,
                    metadata.primary_metric_value
                )
            
            # Emit TRAIN_RUN_FINISHED
            primary_metric_name = metrics_dict["primary_metric_name"]
            primary_metric_value = metrics_dict["metrics_dict"][primary_metric_name.replace("_", "")]
            
            self.emit("TRAIN_RUN_FINISHED", TrainRunFinished(
                run_id=self.current_run_id,
                status="success",
                final_metrics={
                    "task_type": task_type,
                    "primary": {
                        "name": primary_metric_name,
                        "split": "test",
                        "value": primary_metric_value
                    },
                    "metrics": metrics_dict["metrics_list"]
                }
            ).model_dump())
            
            return {
                "run_id": self.current_run_id,
                "model_path": str(output_path / "model.joblib"),
                "metrics": metrics_dict,
                "artifacts": artifact_paths,
                "report_path": report_path,
                "metadata": metadata,
            }
            
        except Exception as e:
            # Emit failure event
            self.emit("LOG_LINE", LogLine(
                run_id=self.current_run_id,
                level="error",
                text=f"Training failed: {str(e)}"
            ).model_dump())
            
            self.emit("TRAIN_RUN_FINISHED", TrainRunFinished(
                run_id=self.current_run_id,
                status="failed",
                final_metrics={}
            ).model_dump())
            
            raise
    
    async def _phase_init(
        self, df, target_column, task_type, model_id,
        n_estimators, max_depth, learning_rate
    ):
        """Phase 1: Initialization."""
        # Emit TRAIN_RUN_STARTED
        config = {
            "task_type": task_type,
            "target": target_column,
            "split": {
                "seed": self.random_state,
                "test_size": self.test_size,
                "val_size": self.val_size
            },
            "model": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate
            },
            "train": {
                "steps": n_estimators,  # Each estimator = 1 step
                "epochs": 1
            }
        }
        
        primary_metric = "f1" if task_type == "classification" else "rmse"
        
        self.emit("TRAIN_RUN_STARTED", TrainRunStarted(
            run_id=self.current_run_id,
            model_id=model_id,
            metric_primary=primary_metric,
            config=config
        ).model_dump())
        
        self.emit("LOG_LINE", LogLine(
            run_id=self.current_run_id,
            level="info",
            text=f"Initializing {model_id} training..."
        ).model_dump())
        
        # Emit initial progress
        self.emit("TRAIN_PROGRESS", TrainProgress(
            run_id=self.current_run_id,
            epoch=1,
            epochs=1,
            step=0,
            steps=n_estimators,
            eta_s=None,
            phase="init"
        ).model_dump())
        
        await asyncio.sleep(0.1)  # Small delay for UX
    
    def _split_data(
        self, df: pd.DataFrame, target_column: str, task_type: str
    ) -> Tuple:
        """Split data into train/val/test."""
        # Drop rows with missing target
        df_clean = df.dropna(subset=[target_column])
        
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]
        
        # Stratify for classification
        stratify = y if task_type == "classification" else None
        
        # Train/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Train/val split
        val_size_adjusted = self.val_size / (1 - self.test_size)
        stratify_temp = y_temp if task_type == "classification" else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_temp
        )
        
        self.emit("LOG_LINE", LogLine(
            run_id=self.current_run_id,
            level="info",
            text=f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        ).model_dump())
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    async def _phase_fit(
        self, X_train, y_train, X_val, y_val,
        task_type, preprocessor,
        n_estimators, max_depth, learning_rate
    ):
        """Phase 2: Fit model with REAL streaming callbacks."""
        self.emit("LOG_LINE", LogLine(
            run_id=self.current_run_id,
            level="info",
            text=f"Starting training ({n_estimators} boosting rounds)..."
        ).model_dump())
        
        # Transform data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)
        
        # Create XGBoost model
        if task_type == "classification":
            n_classes = len(np.unique(y_train))
            if n_classes == 2:
                objective = "binary:logistic"
                eval_metric = "logloss"
            else:
                objective = "multi:softprob"
                eval_metric = "mlogloss"
        else:
            objective = "reg:squarederror"
            eval_metric = "rmse"
        
        # Build model with callback support
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            eval_metric=eval_metric,
            random_state=self.random_state,
            callbacks=[self._create_xgboost_callback()]
        ) if task_type == "classification" else xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            eval_metric=eval_metric,
            random_state=self.random_state,
            callbacks=[self._create_xgboost_callback()]
        )
        
        # Fit with evaluation set (enables callbacks)
        model.fit(
            X_train_transformed, y_train,
            eval_set=[(X_train_transformed, y_train), (X_val_transformed, y_val)],
            verbose=False
        )
        
        # Extract training history
        history = {
            "train_loss": model.evals_result()['validation_0'][eval_metric],
            "val_loss": model.evals_result()['validation_1'][eval_metric],
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        return pipeline, history
    
    def _create_xgboost_callback(self):
        """Create XGBoost callback that emits events after each iteration."""
        
        class StreamingCallback(xgb.callback.TrainingCallback):
            """Custom XGBoost callback for event emission."""
            
            def __init__(self, runner):
                self.runner = runner
                self.iteration_start = time.time()
            
            def after_iteration(self, model, epoch, evals_log):
                """Called after each boosting iteration."""
                # Update iteration times
                current_time = time.time()
                iter_time = current_time - self.iteration_start
                self.runner.iteration_times.append(iter_time)
                self.iteration_start = current_time
                
                # Calculate ETA
                if len(self.runner.iteration_times) > 5:
                    avg_time = np.mean(self.runner.iteration_times[-5:])
                    remaining = model.num_boosted_rounds() - epoch - 1
                    eta_s = avg_time * remaining if remaining > 0 else 0.0
                else:
                    eta_s = None
                
                # Emit TRAIN_PROGRESS
                self.runner.emit("TRAIN_PROGRESS", TrainProgress(
                    run_id=self.runner.current_run_id,
                    epoch=1,
                    epochs=1,
                    step=epoch + 1,
                    steps=model.num_boosted_rounds(),
                    eta_s=eta_s,
                    phase="fit"
                ).model_dump())
                
                # Emit METRIC_SCALAR (REAL values from XGBoost)
                for eval_name, metrics in evals_log.items():
                    split = "train" if eval_name == "validation_0" else "val"
                    for metric_name, values in metrics.items():
                        if values:  # Check if values list is not empty
                            metric_value = values[-1]  # Last value
                            
                            self.runner.emit("METRIC_SCALAR", MetricScalar(
                                run_id=self.runner.current_run_id,
                                name=metric_name,
                                split=split,
                                step=epoch + 1,
                                value=float(metric_value)
                            ).model_dump())
                
                # Emit RESOURCE_STATS periodically (every 10 iterations)
                if epoch % 10 == 0:
                    try:
                        import psutil
                        cpu_pct = psutil.cpu_percent(interval=None)
                        ram_mb = psutil.Process().memory_info().rss / 1024 / 1024
                        avg_time = np.mean(self.runner.iteration_times[-5:]) if len(self.runner.iteration_times) > 5 else 0.1
                        
                        self.runner.emit("RESOURCE_STATS", ResourceStats(
                            run_id=self.runner.current_run_id,
                            cpu_pct=cpu_pct,
                            ram_mb=ram_mb,
                            gpu_pct=0.0,  # No GPU for sklearn/xgboost CPU
                            vram_mb=0.0,
                            step_per_sec=1.0 / avg_time if avg_time > 0 else 0.0
                        ).model_dump())
                    except ImportError:
                        pass  # psutil not available
                
                return False  # Continue training
        
        return StreamingCallback(self)
    
    async def _phase_evaluate(
        self, pipeline, X_test, y_test, task_type
    ):
        """Phase 3: Evaluate on test set."""
        self.emit("TRAIN_PROGRESS", TrainProgress(
            run_id=self.current_run_id,
            epoch=1,
            epochs=1,
            step=pipeline.named_steps['model'].n_estimators,
            steps=pipeline.named_steps['model'].n_estimators,
            eta_s=0,
            phase="eval"
        ).model_dump())
        
        self.emit("LOG_LINE", LogLine(
            run_id=self.current_run_id,
            level="info",
            text="Training complete. Evaluating on test set..."
        ).model_dump())
        
        # Get predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test) if task_type == "classification" else None
        
        # Compute metrics
        primary_metric_name, metrics_list, metrics_dict = compute_all_metrics(
            y_test.values, y_pred, task_type, y_pred_proba
        )
        
        # Emit final metrics
        for metric in metrics_list:
            self.emit("METRIC_SCALAR", MetricScalar(
                run_id=self.current_run_id,
                name=metric["name"],
                split="test",
                step=pipeline.named_steps['model'].n_estimators,
                value=metric["value"]
            ).model_dump())
        
        return {
            "primary_metric_name": primary_metric_name,
            "metrics_list": metrics_list,
            "metrics_dict": metrics_dict,
            "y_test": y_test,
            "y_pred": y_pred,
        }
    
    async def _phase_finalize(
        self, pipeline, X_test, y_test, metrics_dict, task_type,
        output_path, history
    ):
        """Phase 4: Save model and generate artifacts."""
        self.emit("TRAIN_PROGRESS", TrainProgress(
            run_id=self.current_run_id,
            epoch=1,
            epochs=1,
            step=pipeline.named_steps['model'].n_estimators,
            steps=pipeline.named_steps['model'].n_estimators,
            eta_s=0,
            phase="finalize"
        ).model_dump())
        
        self.emit("LOG_LINE", LogLine(
            run_id=self.current_run_id,
            level="info",
            text="Generating artifacts..."
        ).model_dump())
        
        artifact_paths = {}
        
        # Remove callbacks before saving (can't pickle local classes)
        pipeline.named_steps['model'].set_params(callbacks=None)
        
        # Save model
        model_path = output_path / "model.joblib"
        joblib.dump(pipeline, model_path)
        artifact_paths["model"] = str(model_path)
        
        # Generate confusion matrix (classification)
        if task_type == "classification" and "confusion_matrix" in metrics_dict["metrics_dict"]:
            cm_path = output_path / "confusion_matrix.png"
            generate_confusion_matrix_plot(
                metrics_dict["metrics_dict"]["confusion_matrix"],
                output_path=str(cm_path)
            )
            artifact_paths["confusion_matrix"] = str(cm_path)
            
            self.emit("CONFUSION_MATRIX_READY", {
                "asset_url": f"/api/assets/{cm_path}"
            })
        
        # Generate residuals plot (regression)
        if task_type == "regression":
            residuals_path = output_path / "residuals.png"
            generate_residuals_plot(
                metrics_dict["y_test"].values,
                metrics_dict["y_pred"],
                output_path=str(residuals_path)
            )
            artifact_paths["residuals"] = str(residuals_path)
            
            self.emit("RESIDUALS_PLOT_READY", {
                "asset_url": f"/api/assets/{residuals_path}"
            })
        
        # Generate feature importance
        if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            importance_path = output_path / "feature_importance.png"
            
            # Get feature names from preprocessor
            try:
                feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(len(pipeline.named_steps['model'].feature_importances_))]
            
            generate_feature_importance_plot(
                feature_names,
                pipeline.named_steps['model'].feature_importances_,
                output_path=str(importance_path)
            )
            artifact_paths["feature_importance"] = str(importance_path)
            
            self.emit("FEATURE_IMPORTANCE_READY", {
                "asset_url": f"/api/assets/{importance_path}"
            })
        
        # Generate training curves
        curves_path = output_path / "training_curves.png"
        generate_training_curves(history, output_path=str(curves_path))
        artifact_paths["training_curves"] = str(curves_path)
        
        self.emit("LOG_LINE", LogLine(
            run_id=self.current_run_id,
            level="info",
            text=f"Training complete! Model saved to {model_path}"
        ).model_dump())
        
        return artifact_paths


# Legacy compatibility
class TrainerAgent(TrainingRunner):
    """Legacy wrapper for backward compatibility."""
    pass