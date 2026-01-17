"""
Report Generator Module (Task 5.3 - Phase 3)

Generates comprehensive training reports in JSON format.
Creates detailed summaries of model performance and configuration.

Output:
    data/projects/{project_id}/runs/{run_id}/report.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model_registry import ModelMetadata, ModelRegistry


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """
    Generates comprehensive training reports.
    
    Features:
    - Detailed JSON reports with all metrics
    - Training configuration summary
    - Artifact inventory
    - Comparison reports for multiple runs
    - Export-ready format
    
    Usage:
        generator = ReportGenerator()
        
        # Generate report
        report_path = generator.generate_report_json(
            run_id="abc123",
            project_id="project_1",
            metadata=model_metadata
        )
        
        # Compare multiple runs
        comparison = generator.generate_comparison_report(
            project_id="project_1",
            run_ids=["abc", "def", "xyz"]
        )
    """
    
    def __init__(self, base_dir: str = "data/projects"):
        """
        Initialize report generator.
        
        Args:
            base_dir: Base directory for project data
        """
        self.base_dir = Path(base_dir)
        self.registry = ModelRegistry(base_dir=str(base_dir))
    
    def _get_report_path(self, project_id: str, run_id: str) -> Path:
        """Get path to report.json."""
        return self.base_dir / project_id / "runs" / run_id / "report.json"
    
    def generate_report_json(
        self,
        run_id: str,
        project_id: str,
        metadata: ModelMetadata,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive training report.
        
        Args:
            run_id: Run identifier
            project_id: Project identifier
            metadata: ModelMetadata instance
            additional_info: Additional information to include
            
        Returns:
            Path to generated report.json
        """
        # Build report structure
        report = {
            "report_version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            
            # Run identity
            "run_info": {
                "run_id": run_id,
                "project_id": project_id,
                "timestamp": metadata.timestamp,
                "tags": metadata.tags,
                "notes": metadata.notes
            },
            
            # Model information
            "model": {
                "id": metadata.model_id,
                "family": metadata.model_family,
                "task_type": metadata.task_type,
                "hyperparameters": metadata.hyperparameters
            },
            
            # Performance metrics
            "metrics": {
                "primary": {
                    "name": metadata.primary_metric_name,
                    "value": metadata.primary_metric_value
                },
                "all_metrics": metadata.metrics,
                "metric_descriptions": self._get_metric_descriptions(
                    metadata.task_type
                )
            },
            
            # Training details
            "training": {
                "duration_seconds": metadata.training_duration_seconds,
                "config": metadata.training_config,
                "data_splits": {
                    "train_samples": metadata.n_train_samples,
                    "val_samples": metadata.n_val_samples,
                    "test_samples": metadata.n_test_samples,
                    "total_samples": (
                        metadata.n_train_samples +
                        metadata.n_val_samples +
                        metadata.n_test_samples
                    )
                },
                "features": {
                    "n_features": metadata.n_features,
                    "feature_names": metadata.feature_names[:20]  # First 20
                }
            },
            
            # Artifacts
            "artifacts": {
                "model_path": metadata.artifact_paths.get("model", ""),
                "plots": {
                    k: v for k, v in metadata.artifact_paths.items()
                    if k != "model"
                },
                "total_size_mb": self._calculate_total_size(
                    project_id, run_id
                )
            },
            
            # System information
            "system": {
                "python_version": metadata.python_version,
                "xgboost_version": metadata.xgboost_version,
                "sklearn_version": metadata.sklearn_version
            },
            
            # Recommendations
            "recommendations": self._generate_recommendations(metadata)
        }
        
        # Add additional info if provided
        if additional_info:
            report["additional_info"] = additional_info
        
        # Save report
        report_path = self._get_report_path(project_id, run_id)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(report_path)
    
    def _get_metric_descriptions(self, task_type: str) -> Dict[str, str]:
        """Get descriptions for metrics based on task type."""
        if task_type == "classification":
            return {
                "accuracy": "Percentage of correct predictions",
                "precision": "Percentage of positive predictions that are correct",
                "recall": "Percentage of actual positives correctly identified",
                "f1": "Harmonic mean of precision and recall",
                "confusion_matrix": "Matrix showing prediction vs actual counts"
            }
        else:  # regression
            return {
                "rmse": "Root Mean Squared Error - lower is better",
                "mae": "Mean Absolute Error - lower is better",
                "r2": "R-squared score - higher is better (max 1.0)",
                "residuals": "Differences between predictions and actual values"
            }
    
    def _calculate_total_size(self, project_id: str, run_id: str) -> float:
        """Calculate total size of all artifacts in MB."""
        try:
            return self.registry.get_model_size_mb(run_id, project_id)
        except:
            return 0.0
    
    def _generate_recommendations(
        self,
        metadata: ModelMetadata
    ) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if metadata.task_type == "classification":
            # Check for imbalanced precision/recall
            precision = metadata.metrics.get("precision", 0)
            recall = metadata.metrics.get("recall", 0)
            
            if precision > recall + 0.1:
                recommendations.append(
                    "Precision is significantly higher than recall. "
                    "Consider adjusting decision threshold or class weights "
                    "to balance precision and recall."
                )
            elif recall > precision + 0.1:
                recommendations.append(
                    "Recall is significantly higher than precision. "
                    "This may indicate the model is too aggressive in "
                    "positive predictions."
                )
            
            # Check overall performance
            f1 = metadata.metrics.get("f1", 0)
            if f1 < 0.7:
                recommendations.append(
                    "F1 score is below 0.7. Consider: "
                    "(1) Collecting more data, "
                    "(2) Feature engineering, "
                    "(3) Trying different models, "
                    "(4) Hyperparameter tuning."
                )
            elif f1 > 0.95:
                recommendations.append(
                    "Exceptionally high F1 score (>0.95). "
                    "Verify there's no data leakage and test set is representative."
                )
        
        else:  # regression
            r2 = metadata.metrics.get("r2", 0)
            
            if r2 < 0.5:
                recommendations.append(
                    "R² score is below 0.5, indicating poor model fit. "
                    "Consider: (1) Adding more relevant features, "
                    "(2) Removing outliers, (3) Trying non-linear models."
                )
            elif r2 > 0.99:
                recommendations.append(
                    "Extremely high R² (>0.99). "
                    "Check for data leakage or overfitting."
                )
        
        # Duration recommendations
        if metadata.training_duration_seconds > 300:  # 5 minutes
            recommendations.append(
                "Training took over 5 minutes. "
                "For faster iteration, consider reducing n_estimators "
                "or max_depth during hyperparameter search."
            )
        
        if not recommendations:
            recommendations.append(
                "Model performance looks good! "
                "Consider testing on additional validation sets "
                "to ensure generalization."
            )
        
        return recommendations
    
    def generate_comparison_report(
        self,
        project_id: str,
        run_ids: List[str],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comparison report for multiple runs.
        
        Args:
            project_id: Project identifier
            run_ids: List of run IDs to compare
            output_path: Optional path to save comparison (JSON)
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "report_type": "comparison",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "project_id": project_id,
            "runs": [],
            "metric_comparison": {},
            "winner": None
        }
        
        all_metrics = set()
        best_primary_value = float('-inf')
        best_run = None
        
        # Collect data from all runs
        for run_id in run_ids:
            try:
                metadata = self.registry.get_metadata(run_id, project_id)
                
                run_summary = {
                    "run_id": run_id,
                    "model_family": metadata.model_family,
                    "timestamp": metadata.timestamp,
                    "metrics": metadata.metrics,
                    "primary_metric": {
                        "name": metadata.primary_metric_name,
                        "value": metadata.primary_metric_value
                    },
                    "training_duration_seconds": metadata.training_duration_seconds
                }
                
                comparison["runs"].append(run_summary)
                all_metrics.update(metadata.metrics.keys())
                
                # Track best
                if metadata.primary_metric_value > best_primary_value:
                    best_primary_value = metadata.primary_metric_value
                    best_run = run_id
                
            except FileNotFoundError:
                comparison["runs"].append({
                    "run_id": run_id,
                    "error": "Not found"
                })
        
        # Build metric comparison table
        for metric_name in sorted(all_metrics):
            comparison["metric_comparison"][metric_name] = []
            
            for run_summary in comparison["runs"]:
                if "error" not in run_summary:
                    value = run_summary["metrics"].get(metric_name, None)
                    comparison["metric_comparison"][metric_name].append({
                        "run_id": run_summary["run_id"],
                        "model": run_summary["model_family"],
                        "value": value
                    })
        
        # Identify winner
        comparison["winner"] = {
            "run_id": best_run,
            "primary_metric_value": best_primary_value
        }
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(comparison, f, indent=2)
        
        return comparison
    
    def load_report(self, run_id: str, project_id: str) -> Dict[str, Any]:
        """
        Load existing report.
        
        Args:
            run_id: Run identifier
            project_id: Project identifier
            
        Returns:
            Report dictionary
            
        Raises:
            FileNotFoundError: If report doesn't exist
        """
        report_path = self._get_report_path(project_id, run_id)
        
        if not report_path.exists():
            raise FileNotFoundError(f"Report not found: {report_path}")
        
        with open(report_path, 'r') as f:
            return json.load(f)
    
    def generate_summary_text(self, metadata: ModelMetadata) -> str:
        """
        Generate human-readable summary text.
        
        Args:
            metadata: ModelMetadata instance
            
        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            f"Training Report: {metadata.model_id}",
            "=" * 60,
            f"Run ID: {metadata.run_id}",
            f"Timestamp: {metadata.timestamp}",
            f"Task Type: {metadata.task_type.capitalize()}",
            f"Model Family: {metadata.model_family}",
            "",
            "Performance Metrics:",
            "-" * 60
        ]
        
        # Add metrics
        for metric_name, value in sorted(metadata.metrics.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f"  {metric_name:20s}: {value:.4f}")
        
        lines.extend([
            "",
            "Training Details:",
            "-" * 60,
            f"  Duration: {metadata.training_duration_seconds:.1f} seconds",
            f"  Train samples: {metadata.n_train_samples}",
            f"  Val samples: {metadata.n_val_samples}",
            f"  Test samples: {metadata.n_test_samples}",
            f"  Features: {metadata.n_features}",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)
