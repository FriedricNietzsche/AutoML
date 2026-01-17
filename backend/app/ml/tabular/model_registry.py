"""
Model Registry Module (Task 5.3 - Phase 1)

Manages trained models with metadata, versioning, and persistence.
Provides CRUD operations for model lifecycle management.

Storage structure:
    data/projects/{project_id}/runs/{run_id}/
        ├── model.joblib
        ├── metadata.json
        ├── confusion_matrix.png
        ├── feature_importance.png
        └── training_curves.png
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Models
# ============================================================================

class ModelMetadata(BaseModel):
    """Metadata for a trained model run."""
    
    # Identity
    run_id: str
    project_id: str
    timestamp: str  # ISO 8601 format
    
    # Model info
    model_id: str  # e.g., "xgboost_classifier"
    model_family: str  # "XGBoost", "RandomForest", etc.
    task_type: str  # "classification" or "regression"
    
    # Training configuration
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]  # splits, seed, etc.
    
    # Performance metrics
    metrics: Dict[str, Any]  # All test metrics (can include arrays like confusion_matrix)
    primary_metric_name: str
    primary_metric_value: float
    
    # Artifacts
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
    
    # Data info
    n_train_samples: int = 0
    n_val_samples: int = 0
    n_test_samples: int = 0
    n_features: int = 0
    feature_names: List[str] = Field(default_factory=list)
    
    # Optional metadata
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    training_duration_seconds: float = 0.0
    
    # System info
    python_version: str = ""
    xgboost_version: str = ""
    sklearn_version: str = ""


class ModelSummary(BaseModel):
    """Lightweight model summary for listing."""
    run_id: str
    timestamp: str
    model_family: str
    task_type: str
    primary_metric_name: str
    primary_metric_value: float
    tags: List[str]


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """
    Manages trained models with metadata and artifacts.
    
    Provides:
    - Save/load models with metadata
    - List and filter models
    - Find best models by metric
    - Delete old models
    - Model comparison
    
    Usage:
        registry = ModelRegistry(base_dir="data/projects")
        
        # Save model
        registry.save_model_with_metadata(
            run_id="run_abc123",
            project_id="project_1",
            pipeline=trained_pipeline,
            metadata=model_metadata
        )
        
        # Load model
        pipeline = registry.load_model("run_abc123", "project_1")
        
        # Get best model
        best = registry.get_best_model("project_1", "f1")
    """
    
    def __init__(self, base_dir: str = "data/projects"):
        """
        Initialize model registry.
        
        Args:
            base_dir: Base directory for project data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_run_dir(self, project_id: str, run_id: str) -> Path:
        """Get directory for a specific run."""
        return self.base_dir / project_id / "runs" / run_id
    
    def _get_metadata_path(self, project_id: str, run_id: str) -> Path:
        """Get path to metadata.json for a run."""
        return self._get_run_dir(project_id, run_id) / "metadata.json"
    
    def _get_model_path(self, project_id: str, run_id: str) -> Path:
        """Get path to model.joblib for a run."""
        return self._get_run_dir(project_id, run_id) / "model.joblib"
    
    def save_model_with_metadata(
        self,
        run_id: str,
        project_id: str,
        pipeline: Any,
        metadata: ModelMetadata
    ) -> str:
        """
        Save trained model pipeline with metadata.
        
        Args:
            run_id: Unique run identifier
            project_id: Project identifier
            pipeline: Trained sklearn Pipeline
            metadata: ModelMetadata instance
            
        Returns:
            Path to saved model (str)
        """
        # Create run directory
        run_dir = self._get_run_dir(project_id, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model pipeline
        model_path = self._get_model_path(project_id, run_id)
        joblib.dump(pipeline, model_path)
        
        # Save metadata
        metadata_path = self._get_metadata_path(project_id, run_id)
        with open(metadata_path, 'w') as f:
            json.dump(metadata.model_dump(), f, indent=2)
        
        return str(model_path)
    
    def load_model(self, run_id: str, project_id: str) -> Any:
        """
        Load trained model pipeline.
        
        Args:
            run_id: Run identifier
            project_id: Project identifier
            
        Returns:
            Loaded sklearn Pipeline
            
        Raises:
            FileNotFoundError: If model doesn't exist
        """
        model_path = self._get_model_path(project_id, run_id)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )
        
        return joblib.load(model_path)
    
    def get_metadata(self, run_id: str, project_id: str) -> ModelMetadata:
        """
        Get metadata for a run.
        
        Args:
            run_id: Run identifier
            project_id: Project identifier
            
        Returns:
            ModelMetadata instance
            
        Raises:
            FileNotFoundError: If metadata doesn't exist
        """
        metadata_path = self._get_metadata_path(project_id, run_id)
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}"
            )
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        return ModelMetadata(**data)
    
    def list_runs(
        self,
        project_id: str,
        task_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "timestamp",
        descending: bool = True
    ) -> List[ModelSummary]:
        """
        List all runs for a project with optional filtering.
        
        Args:
            project_id: Project identifier
            task_type: Filter by task type (optional)
            tags: Filter by tags (optional, match any)
            sort_by: Sort field (timestamp, primary_metric_value)
            descending: Sort order
            
        Returns:
            List of ModelSummary instances
        """
        runs_dir = self.base_dir / project_id / "runs"
        
        if not runs_dir.exists():
            return []
        
        summaries = []
        
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            try:
                metadata = self.get_metadata(run_dir.name, project_id)
                
                # Apply filters
                if task_type and metadata.task_type != task_type:
                    continue
                
                if tags and not any(tag in metadata.tags for tag in tags):
                    continue
                
                # Create summary
                summary = ModelSummary(
                    run_id=metadata.run_id,
                    timestamp=metadata.timestamp,
                    model_family=metadata.model_family,
                    task_type=metadata.task_type,
                    primary_metric_name=metadata.primary_metric_name,
                    primary_metric_value=metadata.primary_metric_value,
                    tags=metadata.tags
                )
                summaries.append(summary)
                
            except Exception as e:
                # Skip corrupted metadata
                print(f"Warning: Skipping run {run_dir.name}: {e}")
                continue
        
        # Sort
        if sort_by == "timestamp":
            summaries.sort(key=lambda x: x.timestamp, reverse=descending)
        elif sort_by == "primary_metric_value":
            summaries.sort(key=lambda x: x.primary_metric_value, reverse=descending)
        
        return summaries
    
    def get_best_model(
        self,
        project_id: str,
        metric_name: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """
        Get best model by metric value.
        
        Args:
            project_id: Project identifier
            metric_name: Metric to optimize (uses primary_metric if None)
            task_type: Filter by task type (optional)
            
        Returns:
            ModelMetadata for best model, or None if no models
        """
        summaries = self.list_runs(
            project_id,
            task_type=task_type,
            sort_by="primary_metric_value",
            descending=True  # Assume higher is better (works for accuracy, f1, r2)
        )
        
        if not summaries:
            return None
        
        # If specific metric requested, re-evaluate
        if metric_name:
            best_metadata = None
            best_value = float('-inf')
            
            for summary in summaries:
                metadata = self.get_metadata(summary.run_id, project_id)
                
                if metric_name in metadata.metrics:
                    value = metadata.metrics[metric_name]
                    if value > best_value:
                        best_value = value
                        best_metadata = metadata
            
            return best_metadata
        else:
            # Use primary metric
            return self.get_metadata(summaries[0].run_id, project_id)
    
    def compare_models(
        self,
        project_id: str,
        run_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            project_id: Project identifier
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison dictionary with metrics, configs, etc.
        """
        comparison = {
            "runs": [],
            "metrics_comparison": {},
            "hyperparameters_comparison": {}
        }
        
        all_metric_names = set()
        
        for run_id in run_ids:
            try:
                metadata = self.get_metadata(run_id, project_id)
                
                comparison["runs"].append({
                    "run_id": run_id,
                    "model_family": metadata.model_family,
                    "timestamp": metadata.timestamp,
                    "metrics": metadata.metrics,
                    "hyperparameters": metadata.hyperparameters
                })
                
                all_metric_names.update(metadata.metrics.keys())
                
            except FileNotFoundError:
                continue
        
        # Build metrics comparison table
        for metric_name in all_metric_names:
            comparison["metrics_comparison"][metric_name] = []
            
            for run_info in comparison["runs"]:
                value = run_info["metrics"].get(metric_name, None)
                comparison["metrics_comparison"][metric_name].append({
                    "run_id": run_info["run_id"],
                    "value": value
                })
        
        return comparison
    
    def delete_run(self, run_id: str, project_id: str) -> bool:
        """
        Delete a model run and all its artifacts.
        
        Args:
            run_id: Run identifier
            project_id: Project identifier
            
        Returns:
            True if deleted, False if not found
        """
        run_dir = self._get_run_dir(project_id, run_id)
        
        if not run_dir.exists():
            return False
        
        shutil.rmtree(run_dir)
        return True
    
    def cleanup_old_runs(
        self,
        project_id: str,
        keep_n: int = 10,
        keep_best: bool = True
    ) -> List[str]:
        """
        Delete old runs, keeping only the most recent N.
        
        Args:
            project_id: Project identifier
            keep_n: Number of runs to keep
            keep_best: Always keep best model regardless of age
            
        Returns:
            List of deleted run_ids
        """
        summaries = self.list_runs(
            project_id,
            sort_by="timestamp",
            descending=True
        )
        
        if len(summaries) <= keep_n:
            return []
        
        # Determine which to keep
        to_keep = set(s.run_id for s in summaries[:keep_n])
        
        if keep_best:
            best = self.get_best_model(project_id)
            if best:
                to_keep.add(best.run_id)
        
        # Delete the rest
        deleted = []
        for summary in summaries[keep_n:]:
            if summary.run_id not in to_keep:
                if self.delete_run(summary.run_id, project_id):
                    deleted.append(summary.run_id)
        
        return deleted
    
    def get_model_size_mb(self, run_id: str, project_id: str) -> float:
        """
        Get total size of model and artifacts in MB.
        
        Args:
            run_id: Run identifier
            project_id: Project identifier
            
        Returns:
            Size in megabytes
        """
        run_dir = self._get_run_dir(project_id, run_id)
        
        if not run_dir.exists():
            return 0.0
        
        total_bytes = sum(
            f.stat().st_size
            for f in run_dir.rglob('*')
            if f.is_file()
        )
        
        return total_bytes / (1024 * 1024)


# ============================================================================
# Utility Functions
# ============================================================================

def create_metadata_from_training(
    run_id: str,
    project_id: str,
    model_id: str,
    model_family: str,
    task_type: str,
    metrics: Dict[str, float],
    primary_metric_name: str,
    hyperparameters: Dict[str, Any],
    training_config: Dict[str, Any],
    artifact_paths: Dict[str, str],
    **kwargs
) -> ModelMetadata:
    """
    Helper to create ModelMetadata from training results.
    
    Args:
        run_id: Unique run identifier
        project_id: Project identifier
        model_id: Model identifier
        model_family: Model family name
        task_type: Classification or regression
        metrics: All computed metrics
        primary_metric_name: Name of primary metric
        hyperparameters: Model hyperparameters
        training_config: Training configuration
        artifact_paths: Paths to generated artifacts
        **kwargs: Additional metadata fields
        
    Returns:
        ModelMetadata instance
    """
    import sys
    import sklearn
    import xgboost as xgb
    
    return ModelMetadata(
        run_id=run_id,
        project_id=project_id,
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_id=model_id,
        model_family=model_family,
        task_type=task_type,
        metrics=metrics,
        primary_metric_name=primary_metric_name,
        primary_metric_value=metrics.get(primary_metric_name.replace("_", ""), 0.0),
        hyperparameters=hyperparameters,
        training_config=training_config,
        artifact_paths=artifact_paths,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        xgboost_version=xgb.__version__,
        sklearn_version=sklearn.__version__,
        **kwargs
    )
