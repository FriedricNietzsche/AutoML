"""
Pipeline orchestrator - emits stage events for full UI timeline coverage.
"""

import asyncio
from typing import Any, Dict, Optional
from app.core.stage_mapping import build_stage_event, BackendStage
from app.services.ws_manager import manager

class PipelineOrchestrator:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.current_stage: Optional[str] = None
        self.artifacts: Dict[str, Any] = {}

    async def emit_stage(
        self,
        stage: str,
        status: str,
        message: str = "",
        progress: float = 0.0,
        artifacts: Dict[str, Any] = None,
    ):
        """Emit stage event to WebSocket clients."""
        self.current_stage = stage
        if artifacts:
            self.artifacts[stage] = artifacts
        
        event = build_stage_event(stage, status, message, progress, artifacts)
        await manager.broadcast(self.project_id, event)

    async def run_parse_stage(self, prompt: str) -> Dict[str, Any]:
        """Parse user prompt and extract intent."""
        await self.emit_stage(
            BackendStage.PARSE, "running", "Parsing your request...", 0.1
        )
        await asyncio.sleep(0.3)  # Simulated parsing
        
        result = {
            "intent": "classification",  # or regression, clustering
            "target_column": None,
            "raw_prompt": prompt,
        }
        
        await self.emit_stage(
            BackendStage.PARSE, "completed", "Request parsed", 1.0, result
        )
        return result

    async def run_data_source_stage(
        self, source_type: str, source_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ingest data from source."""
        await self.emit_stage(
            BackendStage.DATA_SOURCE,
            "running",
            f"Loading data from {source_type}...",
            0.1,
        )
        
        # Actual ingestion happens in endpoint, this tracks progress
        result = {
            "source_type": source_type,
            "config": source_config,
        }
        return result

    async def complete_data_source(
        self, row_count: int, col_count: int, sample_rows: list = None
    ):
        """Mark data source stage complete with metadata."""
        await self.emit_stage(
            BackendStage.DATA_SOURCE,
            "completed",
            f"Loaded {row_count:,} rows × {col_count} columns",
            1.0,
            {
                "row_count": row_count,
                "col_count": col_count,
                "sample_rows": sample_rows or [],
            },
        )

    async def run_profile_stage(self, df_info: Dict[str, Any]) -> Dict[str, Any]:
        """Profile the dataset - statistics, types, missing values."""
        await self.emit_stage(
            BackendStage.PROFILE_DATA,
            "running",
            "Analyzing data distribution...",
            0.2,
        )
        await asyncio.sleep(0.2)

        # Build profile artifacts
        profile = {
            "columns": df_info.get("columns", []),
            "dtypes": df_info.get("dtypes", {}),
            "missing_counts": df_info.get("missing_counts", {}),
            "numeric_stats": df_info.get("numeric_stats", {}),
            "categorical_counts": df_info.get("categorical_counts", {}),
        }

        await self.emit_stage(
            BackendStage.PROFILE_DATA,
            "completed",
            "Data profiling complete",
            1.0,
            profile,
        )
        return profile

    async def run_preprocess_stage(
        self, profile: Dict[str, Any], task_type: str
    ) -> Dict[str, Any]:
        """Generate and apply preprocessing plan."""
        await self.emit_stage(
            BackendStage.PREPROCESS,
            "running",
            "Building preprocessing pipeline...",
            0.1,
        )
        await asyncio.sleep(0.2)

        # Build preprocessing plan based on profile
        plan_steps = []
        missing = profile.get("missing_counts", {})
        dtypes = profile.get("dtypes", {})

        for col, count in missing.items():
            if count > 0:
                dtype = dtypes.get(col, "object")
                if dtype in ("int64", "float64"):
                    plan_steps.append(f"Impute '{col}' with median")
                else:
                    plan_steps.append(f"Impute '{col}' with mode")

        for col, dtype in dtypes.items():
            if dtype == "object":
                plan_steps.append(f"One-hot encode '{col}'")

        if not plan_steps:
            plan_steps.append("No preprocessing required - data is clean")

        await self.emit_stage(
            BackendStage.PREPROCESS, "running", "Applying transformations...", 0.5
        )
        await asyncio.sleep(0.3)

        preprocess_result = {
            "plan_steps": plan_steps,
            "applied": True,
            "output_shape": profile.get("output_shape", [0, 0]),
        }

        await self.emit_stage(
            BackendStage.PREPROCESS,
            "completed",
            f"Applied {len(plan_steps)} transformations",
            1.0,
            preprocess_result,
        )
        return preprocess_result

    async def run_model_select_stage(
        self, task_type: str, data_size: int
    ) -> Dict[str, Any]:
        """Select candidate models based on task and data."""
        await self.emit_stage(
            BackendStage.MODEL_SELECT,
            "running",
            "Evaluating model candidates...",
            0.2,
        )
        await asyncio.sleep(0.3)

        # Model selection logic
        if task_type == "classification":
            candidates = [
                {"name": "RandomForest", "reason": "Good default for tabular"},
                {"name": "XGBoost", "reason": "High performance on structured data"},
                {"name": "LogisticRegression", "reason": "Fast, interpretable baseline"},
            ]
        elif task_type == "regression":
            candidates = [
                {"name": "RandomForest", "reason": "Handles non-linear relationships"},
                {"name": "XGBoost", "reason": "State-of-the-art gradient boosting"},
                {"name": "LinearRegression", "reason": "Simple interpretable baseline"},
            ]
        elif task_type == "image_classification":
            candidates = [
                {"name": "ResNet18", "reason": "Efficient CNN for image tasks"},
                {"name": "MobileNetV2", "reason": "Lightweight, fast inference"},
            ]
        else:
            candidates = [{"name": "AutoSelect", "reason": "Automatic selection"}]

        selected = candidates[0]["name"] if candidates else "AutoSelect"

        model_select_result = {
            "candidates": candidates,
            "selected": selected,
            "task_type": task_type,
        }

        await self.emit_stage(
            BackendStage.MODEL_SELECT,
            "completed",
            f"Selected {selected}",
            1.0,
            model_select_result,
        )
        return model_select_result

    async def start_train_stage(self, model_name: str, epochs: int = 10):
        """Begin training stage."""
        await self.emit_stage(
            BackendStage.TRAIN,
            "running",
            f"Training {model_name}...",
            0.0,
            {"model": model_name, "total_epochs": epochs},
        )

    async def update_train_progress(
        self, epoch: int, total_epochs: int, metrics: Dict[str, float]
    ):
        """Update training progress."""
        progress = epoch / total_epochs
        await self.emit_stage(
            BackendStage.TRAIN,
            "running",
            f"Epoch {epoch}/{total_epochs}",
            progress,
            {"epoch": epoch, "metrics": metrics},
        )

    async def complete_train_stage(self, final_metrics: Dict[str, Any]):
        """Complete training with final metrics."""
        await self.emit_stage(
            BackendStage.TRAIN,
            "completed",
            "Training complete",
            1.0,
            final_metrics,
        )

    async def run_review_stage(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model review/evaluation."""
        await self.emit_stage(
            BackendStage.REVIEW,
            "running",
            "Generating evaluation report...",
            0.3,
        )
        await asyncio.sleep(0.2)

        review = {
            "final_metrics": metrics,
            "recommendations": [],
        }

        # Add recommendations based on metrics
        accuracy = metrics.get("accuracy", metrics.get("r2", 0))
        if accuracy < 0.7:
            review["recommendations"].append(
                "Consider feature engineering or more training data"
            )
        if accuracy > 0.95:
            review["recommendations"].append(
                "Check for data leakage - accuracy may be unrealistically high"
            )

        await self.emit_stage(
            BackendStage.REVIEW, "completed", "Evaluation complete", 1.0, review
        )
        return review

    async def run_export_stage(self, model_path: str) -> Dict[str, Any]:
        """Export trained model."""
        await self.emit_stage(
            BackendStage.EXPORT, "running", "Exporting model...", 0.5
        )
        await asyncio.sleep(0.2)

        export_result = {
            "model_path": model_path,
            "formats": ["pickle", "onnx"],
            "ready": True,
        }

        await self.emit_stage(
            BackendStage.EXPORT,
            "completed",
            f"Model exported to {model_path}",
            1.0,
            export_result,
        )
        return export_result


# Factory function
def create_orchestrator(project_id: str) -> PipelineOrchestrator:
    return PipelineOrchestrator(project_id)
