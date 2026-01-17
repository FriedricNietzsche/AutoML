"""
Data Profiling Module (Task 4.1)

Analyzes datasets to compute summary statistics, detect data quality issues,
and generate profiling assets (missingness tables, distribution plots).

Emits events:
  - PROFILE_PROGRESS: {phase, pct}
  - PROFILE_SUMMARY: {n_rows, n_cols, missing_pct, types_breakdown, warnings:[]}
  - MISSINGNESS_TABLE_READY: {asset_url}
  - TARGET_DISTRIBUTION_READY: {asset_url}

Contract: docs/FRONTEND_BACKEND_CONTRACT.md
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server-side plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    plt = None
    sns = None


class ProfileSummary(BaseModel):
    """Contract payload for PROFILE_SUMMARY event."""
    
    n_rows: int = Field(..., description="Total number of rows")
    n_cols: int = Field(..., description="Total number of columns")
    missing_pct: float = Field(..., description="Overall percentage of missing values")
    types_breakdown: Dict[str, int] = Field(
        ...,
        description="Count of columns by type: numeric, categorical, datetime, other"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Data quality warnings"
    )


class ProfileProgress(BaseModel):
    """Contract payload for PROFILE_PROGRESS event."""
    
    phase: str = Field(..., description="Current profiling phase")
    pct: float = Field(..., description="Progress percentage (0-100)")


class DataProfiler:
    """
    Profiles tabular datasets for ML pipeline.
    
    Usage:
        profiler = DataProfiler(df, target_column="price", task_type="regression")
        
        # Compute profile summary
        summary = profiler.compute_profile()
        # Returns: {n_rows, n_cols, missing_pct, types_breakdown, warnings}
        
        # Generate assets
        missingness_path = profiler.generate_missingness_table(output_dir="assets/")
        distribution_path = profiler.generate_target_distribution(output_dir="assets/")
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
    ):
        """
        Initialize profiler.
        
        Args:
            df: DataFrame to profile
            target_column: Name of the target column (optional)
            task_type: ML task type (classification/regression/clustering/etc.)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.task_type = (task_type or "").lower() if task_type else None
        
    def compute_profile(self) -> Dict[str, Any]:
        """
        Compute comprehensive profile summary.
        
        Returns:
            PROFILE_SUMMARY payload: {n_rows, n_cols, missing_pct, types_breakdown, warnings}
        """
        n_rows, n_cols = self.df.shape
        
        # Overall missing percentage
        total_cells = n_rows * n_cols
        missing_cells = self.df.isnull().sum().sum()
        missing_pct = round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0
        
        # Type breakdown
        types_breakdown = self._classify_column_types()
        
        # Data quality warnings
        warnings = self._generate_warnings()
        
        summary = ProfileSummary(
            n_rows=n_rows,
            n_cols=n_cols,
            missing_pct=missing_pct,
            types_breakdown=types_breakdown,
            warnings=warnings,
        )
        
        return summary.model_dump()
    
    def _classify_column_types(self) -> Dict[str, int]:
        """
        Classify each column as numeric, categorical, datetime, or other.
        
        Returns:
            Dict with counts: {numeric: 5, categorical: 3, datetime: 1, other: 0}
        """
        types = {"numeric": 0, "categorical": 0, "datetime": 0, "other": 0}
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                types["numeric"] += 1
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                types["datetime"] += 1
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                types["categorical"] += 1
            else:
                types["other"] += 1
        
        return types
    
    def _generate_warnings(self) -> List[str]:
        """
        Detect data quality issues and generate warnings.
        
        Returns:
            List of warning messages
        """
        warnings = []
        n_rows = len(self.df)
        
        # Check each column
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / n_rows) * 100 if n_rows > 0 else 0
            
            # High missing values
            if missing_pct > 50:
                warnings.append(
                    f"Column '{col}' has {missing_pct:.1f}% missing values - consider imputation or removal"
                )
            elif missing_pct > 20:
                warnings.append(
                    f"Column '{col}' has {missing_pct:.1f}% missing values - may need imputation"
                )
            
            # High cardinality categoricals (likely ID columns)
            if pd.api.types.is_object_dtype(self.df[col].dtype):
                n_unique = self.df[col].nunique()
                if n_unique > 0.9 * n_rows:
                    warnings.append(
                        f"Column '{col}' has high cardinality ({n_unique} unique values) - likely an ID column"
                    )
            
            # Constant columns (no variance)
            if self.df[col].nunique() == 1:
                warnings.append(f"Column '{col}' has only 1 unique value - consider removing")
        
        # Target-specific warnings
        if self.target_column and self.target_column in self.df.columns:
            target_missing = self.df[self.target_column].isnull().sum()
            if target_missing > 0:
                warnings.append(
                    f"Target column '{self.target_column}' has {target_missing} missing values - rows will be dropped"
                )
            
            # Classification: check class imbalance
            if self.task_type == "classification":
                value_counts = self.df[self.target_column].value_counts()
                if len(value_counts) > 0:
                    majority_pct = (value_counts.iloc[0] / len(self.df)) * 100
                    if majority_pct > 80:
                        warnings.append(
                            f"Target class imbalance detected: {majority_pct:.1f}% majority class - consider balancing techniques"
                        )
            
            # Regression: check for outliers
            elif self.task_type == "regression":
                if pd.api.types.is_numeric_dtype(self.df[self.target_column].dtype):
                    target_values = self.df[self.target_column].dropna()
                    if len(target_values) > 0:
                        q1, q3 = target_values.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = ((target_values < q1 - 3 * iqr) | (target_values > q3 + 3 * iqr)).sum()
                        if outliers > 0:
                            outlier_pct = (outliers / len(target_values)) * 100
                            if outlier_pct > 5:
                                warnings.append(
                                    f"Target has {outliers} extreme outliers ({outlier_pct:.1f}%) - may need clipping or transformation"
                                )
        
        return warnings
    
    def generate_missingness_table(self, output_dir: str) -> str:
        """
        Generate missingness table as JSON asset.
        
        Args:
            output_dir: Directory to save the JSON file
            
        Returns:
            Path to the saved JSON file (relative to output_dir)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute per-column missingness
        missingness_data = []
        n_rows = len(self.df)
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / n_rows) * 100 if n_rows > 0 else 0.0
            
            missingness_data.append({
                "column": col,
                "dtype": str(self.df[col].dtype),
                "missing_count": int(missing_count),
                "missing_pct": round(missing_pct, 2),
                "present_count": int(n_rows - missing_count),
            })
        
        # Sort by missing percentage (descending)
        missingness_data.sort(key=lambda x: x["missing_pct"], reverse=True)
        
        # Save as JSON
        file_path = output_path / "missingness_table.json"
        with open(file_path, "w") as f:
            json.dump(missingness_data, f, indent=2)
        
        return str(file_path)
    
    def generate_target_distribution(self, output_dir: str) -> Optional[str]:
        """
        Generate target distribution plot as PNG asset.
        
        Args:
            output_dir: Directory to save the PNG file
            
        Returns:
            Path to the saved PNG file, or None if plotting not available or no target
        """
        if plt is None:
            return None
        
        if not self.target_column or self.target_column not in self.df.columns:
            return None
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        target_data = self.df[self.target_column].dropna()
        
        if len(target_data) == 0:
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.task_type == "classification":
            # Bar chart for classification
            value_counts = target_data.value_counts().sort_index()
            value_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
            ax.set_xlabel("Class", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"Target Distribution: {self.target_column}", fontsize=14, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            
            # Add count labels on bars
            for i, v in enumerate(value_counts):
                ax.text(i, v + max(value_counts) * 0.01, str(v), ha="center", va="bottom")
        
        else:  # regression or unknown
            # Histogram for regression
            ax.hist(target_data, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
            ax.set_xlabel(self.target_column, fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(f"Target Distribution: {self.target_column}", fontsize=14, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            
            # Add summary stats as text
            mean_val = target_data.mean()
            median_val = target_data.median()
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
            ax.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median: {median_val:.2f}")
            ax.legend()
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save
        file_path = output_path / "target_distribution.png"
        plt.savefig(file_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        
        return str(file_path)


# Legacy functions for backward compatibility
def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Legacy function - profiles dataset and returns summary.
    
    For new code, use DataProfiler class instead.
    """
    profiler = DataProfiler(df)
    return profiler.compute_profile()


def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Legacy function - detects data types.
    
    For new code, use DataProfiler class instead.
    """
    return {col: str(df[col].dtype) for col in df.columns}