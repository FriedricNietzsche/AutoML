"""
ML Metrics and SHAP Explainability Module
Provides comprehensive model evaluation metrics and SHAP explanations.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    explained_variance_score,
    max_error,
    median_absolute_error,
)
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Try to import SHAP - it's optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.info("SHAP not installed - SHAP explanations will be unavailable")


class MetricsCalculator:
    """Calculate comprehensive ML metrics for classification and regression tasks."""

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        average: str = "weighted",
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for ROC-AUC, etc.)
            labels: Class labels (optional)
            average: Averaging strategy for multi-class ('micro', 'macro', 'weighted')
            
        Returns:
            Dictionary containing all classification metrics
        """
        metrics: Dict[str, Any] = {}
        
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        is_binary = n_classes == 2
        
        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        
        # Precision, Recall, F1
        try:
            metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
            metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        except Exception as e:
            logger.warning(f"Error computing precision/recall/f1: {e}")
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
        
        # Per-class metrics
        try:
            metrics["precision_per_class"] = precision_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist()
            metrics["recall_per_class"] = recall_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist()
            metrics["f1_per_class"] = f1_score(
                y_true, y_pred, average=None, zero_division=0
            ).tolist()
        except Exception as e:
            logger.warning(f"Error computing per-class metrics: {e}")
        
        # Matthews Correlation Coefficient
        try:
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception as e:
            logger.warning(f"Error computing MCC: {e}")
            metrics["mcc"] = 0.0
        
        # Cohen's Kappa
        try:
            metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        except Exception as e:
            logger.warning(f"Error computing Cohen's Kappa: {e}")
            metrics["cohen_kappa"] = 0.0
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Compute per-class stats from confusion matrix
            if is_binary:
                tn, fp, fn, tp = cm.ravel()
                metrics["true_positives"] = int(tp)
                metrics["true_negatives"] = int(tn)
                metrics["false_positives"] = int(fp)
                metrics["false_negatives"] = int(fn)
                # Specificity (true negative rate)
                metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                # Sensitivity (recall / true positive rate) 
                metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error computing confusion matrix: {e}")
            metrics["confusion_matrix"] = []
        
        # Classification report (detailed)
        try:
            report = classification_report(
                y_true, y_pred, 
                target_names=labels if labels else None,
                output_dict=True,
                zero_division=0
            )
            metrics["classification_report"] = report
        except Exception as e:
            logger.warning(f"Error computing classification report: {e}")
        
        # Probability-based metrics (if probabilities are provided)
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            
            # ROC-AUC
            try:
                if is_binary:
                    # For binary classification, use probability of positive class
                    if y_prob.ndim == 2:
                        y_prob_pos = y_prob[:, 1]
                    else:
                        y_prob_pos = y_prob
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
                    
                    # ROC curve data
                    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
                    metrics["roc_curve"] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "thresholds": thresholds.tolist(),
                    }
                    
                    # Precision-Recall curve
                    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
                        y_true, y_prob_pos
                    )
                    metrics["pr_curve"] = {
                        "precision": precision_curve.tolist(),
                        "recall": recall_curve.tolist(),
                        "thresholds": pr_thresholds.tolist() if len(pr_thresholds) > 0 else [],
                    }
                    
                    # Average Precision (Area under PR curve)
                    metrics["average_precision"] = float(average_precision_score(y_true, y_prob_pos))
                    
                else:
                    # Multi-class ROC-AUC
                    metrics["roc_auc"] = float(roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average=average
                    ))
            except Exception as e:
                logger.warning(f"Error computing ROC-AUC: {e}")
                metrics["roc_auc"] = None
            
            # Log Loss
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Error computing log loss: {e}")
        
        # Support (sample count per class)
        metrics["n_samples"] = len(y_true)
        metrics["n_classes"] = n_classes
        metrics["class_labels"] = classes.tolist()
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        metrics["class_distribution"] = {
            str(k): int(v) for k, v in zip(unique, counts)
        }
        
        return metrics

    @staticmethod
    def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing all regression metrics
        """
        metrics: Dict[str, Any] = {}
        
        # Convert to numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Mean Squared Error and RMSE
        mse = mean_squared_error(y_true, y_pred)
        metrics["mse"] = float(mse)
        metrics["rmse"] = float(np.sqrt(mse))
        
        # Mean Absolute Error
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        
        # Median Absolute Error (robust to outliers)
        metrics["median_ae"] = float(median_absolute_error(y_true, y_pred))
        
        # R² Score (coefficient of determination)
        metrics["r2"] = float(r2_score(y_true, y_pred))
        
        # Adjusted R² (if we had n_features, we'd compute this properly)
        # For now, just provide R²
        
        # Explained Variance
        metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))
        
        # Max Error
        metrics["max_error"] = float(max_error(y_true, y_pred))
        
        # Mean Absolute Percentage Error (MAPE)
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics["mape"] = float(mape)
        else:
            metrics["mape"] = None
        
        # Symmetric MAPE (handles zero values better)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() > 0:
            smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
            metrics["smape"] = float(smape)
        else:
            metrics["smape"] = None
        
        # Residuals statistics
        residuals = y_true - y_pred
        metrics["residuals"] = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "median": float(np.median(residuals)),
            "values": residuals.tolist()[:100],  # First 100 for visualization
        }
        
        # Sample statistics
        metrics["n_samples"] = len(y_true)
        metrics["y_true_stats"] = {
            "mean": float(np.mean(y_true)),
            "std": float(np.std(y_true)),
            "min": float(np.min(y_true)),
            "max": float(np.max(y_true)),
        }
        metrics["y_pred_stats"] = {
            "mean": float(np.mean(y_pred)),
            "std": float(np.std(y_pred)),
            "min": float(np.min(y_pred)),
            "max": float(np.max(y_pred)),
        }
        
        return metrics


class SHAPExplainer:
    """Generate SHAP explanations for model interpretability."""
    
    def __init__(self, model: Any, model_type: str = "auto"):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            model_type: Type of model ('tree', 'linear', 'deep', 'auto')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        
    def fit(
        self, 
        X_background: Union[np.ndarray, pd.DataFrame],
        max_samples: int = 100
    ) -> "SHAPExplainer":
        """
        Fit the SHAP explainer with background data.
        
        Args:
            X_background: Background dataset for SHAP
            max_samples: Maximum samples to use for background
            
        Returns:
            self
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - explanations will be empty")
            return self
            
        try:
            # Sample background data if too large
            if len(X_background) > max_samples:
                if isinstance(X_background, pd.DataFrame):
                    X_background = X_background.sample(n=max_samples, random_state=42)
                else:
                    idx = np.random.choice(len(X_background), max_samples, replace=False)
                    X_background = X_background[idx]
            
            # Choose appropriate explainer based on model type
            model_class = self.model.__class__.__name__.lower()
            
            if self.model_type == "tree" or any(x in model_class for x in ["tree", "forest", "xgb", "lgb", "catboost"]):
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "linear" or any(x in model_class for x in ["linear", "logistic", "ridge", "lasso"]):
                self.explainer = shap.LinearExplainer(self.model, X_background)
            else:
                # Use KernelExplainer as fallback (works with any model)
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    shap.sample(X_background, min(50, len(X_background)))
                )
                
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")
            self.explainer = None
            
        return self
    
    def explain(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for given data.
        
        Args:
            X: Data to explain
            max_samples: Maximum samples to explain
            
        Returns:
            Dictionary with SHAP values and summary
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"available": False, "message": "SHAP not available"}
        
        try:
            # Sample if too large
            if len(X) > max_samples:
                if isinstance(X, pd.DataFrame):
                    X = X.sample(n=max_samples, random_state=42)
                else:
                    idx = np.random.choice(len(X), max_samples, replace=False)
                    X = X[idx]
            
            # Get feature names
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-output (e.g., classification)
            if isinstance(shap_values, list):
                # For binary classification, take the positive class
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    # For multi-class, average across classes
                    shap_values = np.mean(np.abs(shap_values), axis=0)
            
            # Global feature importance (mean absolute SHAP)
            global_importance = np.mean(np.abs(shap_values), axis=0)
            importance_ranking = sorted(
                zip(feature_names, global_importance),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Build response
            result = {
                "available": True,
                "feature_names": feature_names,
                "global_importance": {name: float(imp) for name, imp in importance_ranking},
                "importance_ranking": [
                    {"feature": name, "importance": float(imp)}
                    for name, imp in importance_ranking
                ],
                # Include sample SHAP values for visualization
                "sample_shap_values": shap_values[:10].tolist() if len(shap_values) > 0 else [],
                "expected_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') and not isinstance(self.explainer.expected_value, (list, np.ndarray)) else None,
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error computing SHAP values: {e}")
            return {"available": False, "message": str(e)}


def compute_metrics_and_explanations(
    model: Any,
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task_type: str = "classification",
    labels: Optional[List[str]] = None,
    compute_shap: bool = True,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics and SHAP explanations.
    
    Args:
        model: Trained model
        X_train: Training features (for SHAP background)
        X_test: Test features
        y_test: True test labels/values
        y_pred: Predicted labels/values
        y_prob: Predicted probabilities (classification only)
        task_type: 'classification' or 'regression'
        labels: Class labels (classification only)
        compute_shap: Whether to compute SHAP explanations
        
    Returns:
        Dictionary with all metrics and explanations
    """
    result: Dict[str, Any] = {
        "task_type": task_type,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    
    # Compute metrics based on task type
    if task_type == "classification":
        result["metrics"] = MetricsCalculator.classification_metrics(
            y_test, y_pred, y_prob, labels
        )
    else:
        result["metrics"] = MetricsCalculator.regression_metrics(y_test, y_pred)
    
    # Compute SHAP explanations if requested
    if compute_shap and SHAP_AVAILABLE:
        try:
            explainer = SHAPExplainer(model)
            explainer.fit(X_train)
            result["shap"] = explainer.explain(X_test)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            result["shap"] = {"available": False, "message": str(e)}
    else:
        result["shap"] = {"available": False, "message": "SHAP not requested or not available"}
    
    # Feature importance from model (if available)
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if isinstance(X_test, pd.DataFrame):
                feature_names = X_test.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            result["feature_importance"] = sorted(
                [{"feature": name, "importance": float(imp)} 
                 for name, imp in zip(feature_names, importances)],
                key=lambda x: x["importance"],
                reverse=True
            )
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_).flatten()
            if isinstance(X_test, pd.DataFrame):
                feature_names = X_test.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            result["feature_importance"] = sorted(
                [{"feature": name, "importance": float(imp)} 
                 for name, imp in zip(feature_names, coef)],
                key=lambda x: x["importance"],
                reverse=True
            )
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
    
    return result


# Streaming metrics for real-time updates during training
class StreamingMetrics:
    """Track and emit metrics during training for real-time updates."""
    
    def __init__(self, project_id: str, run_id: str):
        self.project_id = project_id
        self.run_id = run_id
        self.history: Dict[str, List[Tuple[int, float]]] = {}
    
    def record(self, metric_name: str, value: float, step: int) -> Dict[str, Any]:
        """
        Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Training step
            
        Returns:
            Event payload for streaming
        """
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append((step, value))
        
        return {
            "run_id": self.run_id,
            "name": metric_name,
            "step": step,
            "value": float(value),
            "history": [(s, v) for s, v in self.history[metric_name][-50:]],  # Last 50 points
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded metrics."""
        summary = {}
        for name, history in self.history.items():
            values = [v for _, v in history]
            summary[name] = {
                "last": values[-1] if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": float(np.mean(values)) if values else None,
                "history": history,
            }
        return summary
