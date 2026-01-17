"""
Visualization Module (Task 5.1 - Part 2)

Generates training artifacts:
- Confusion matrix (classification)
- Residuals plot (regression)
- Feature importance (tree-based models)
- ROC curve (binary classification)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    plt = None
    sns = None


def generate_confusion_matrix_plot(
    confusion_matrix: List[List[int]],
    class_labels: Optional[List[str]] = None,
    output_path: str = None,
) -> Optional[str]:
    """
    Generate confusion matrix heatmap.
    
    Args:
        confusion_matrix: 2D list/array of confusion matrix
        class_labels: Optional list of class names
        output_path: Path to save PNG
        
    Returns:
        Path to saved file, or None if matplotlib unavailable
    """
    if plt is None:
        return None
    
    cm_array = np.array(confusion_matrix)
    n_classes = cm_array.shape[0]
    
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(n_classes)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        cm_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def generate_residuals_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = None,
) -> Optional[str]:
    """
    Generate residuals plot for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save PNG
        
    Returns:
        Path to saved file
    """
    if plt is None:
        return None
    
    residuals = y_true - y_pred
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Residuals histogram
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def generate_feature_importance_plot(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: str = None,
    top_n: int = 20,
) -> Optional[str]:
    """
    Generate feature importance bar chart.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        output_path: Path to save PNG
        top_n: Show top N features
        
    Returns:
        Path to saved file
    """
    if plt is None:
        return None
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances, align='center', color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # Highest importance on top
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def generate_training_curves(
    history: Dict[str, List[float]],
    output_path: str = None,
) -> Optional[str]:
    """
    Generate training curves (loss + metric over iterations).
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_metric', 'val_metric'
        output_path: Path to save PNG
        
    Returns:
        Path to saved file
    """
    if plt is None:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Metric
    if 'train_metric' in history:
        ax2.plot(history['train_metric'], label='Train Metric', linewidth=2)
    if 'val_metric' in history:
        ax2.plot(history['val_metric'], label='Val Metric', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Metric', fontsize=12)
    ax2.set_title('Training Metric', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path
