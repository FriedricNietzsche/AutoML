"""
Metrics Computation Module (Task 5.1 - Part 1)

Computes real evaluation metrics for classification and regression tasks.
Used by TrainingRunner to compute final metrics after model training.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for AUC-ROC)
        
    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix
    """
    # Basic metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    
    # Handle binary vs multiclass
    n_classes = len(np.unique(y_true))
    average = 'binary' if n_classes == 2 else 'weighted'
    
    precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "n_classes": n_classes,
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dict with rmse, mae, r2, residuals
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Residuals for plotting
    residuals = (y_true - y_pred).tolist()
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mse": mse,
        "residuals": residuals[:1000],  # Limit size for event payload
    }


def format_metric_for_event(
    metric_name: str,
    value: float,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Format a single metric for METRIC_SCALAR event.
    
    Args:
        metric_name: Name of metric (accuracy, f1, rmse, etc.)
        value: Metric value
        split: train/val/test
        
    Returns:
        Dict formatted for contract
    """
    return {
        "name": metric_name,
        "split": split,
        "value": round(value, 6),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_pred_proba: np.ndarray = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Compute all relevant metrics for a task type.
    
    Args:
        y_true: True labels/values
        y_pred: Predictions
        task_type: 'classification' or 'regression'
        y_pred_proba: Predicted probabilities (classification only)
        
    Returns:
        (primary_metric_name, list_of_metric_dicts)
    """
    if task_type == "classification":
        metrics_dict = compute_classification_metrics(y_true, y_pred, y_pred_proba)
        
        # Format for event emission
        metrics_list = [
            format_metric_for_event("accuracy", metrics_dict["accuracy"]),
            format_metric_for_event("precision", metrics_dict["precision"]),
            format_metric_for_event("recall", metrics_dict["recall"]),
            format_metric_for_event("f1", metrics_dict["f1"]),
        ]
        
        primary_metric = "f1"  # F1 is primary for classification
        
        return primary_metric, metrics_list, metrics_dict
    
    else:  # regression
        metrics_dict = compute_regression_metrics(y_true, y_pred)
        
        metrics_list = [
            format_metric_for_event("rmse", metrics_dict["rmse"]),
            format_metric_for_event("mae", metrics_dict["mae"]),
            format_metric_for_event("r2", metrics_dict["r2"]),
        ]
        
        primary_metric = "rmse"  # RMSE is primary for regression
        
        return primary_metric, metrics_list, metrics_dict
