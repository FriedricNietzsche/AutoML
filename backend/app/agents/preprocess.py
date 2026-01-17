"""
Preprocessing Planner Agent (Task 4.2)

Creates preprocessing pipeline plans based on data profiling results.
Emits:
  - PREPROCESS_PLAN: {steps:[...]}
  - SPLIT_SUMMARY: {train_rows, val_rows, test_rows, stratified, seed}

Contract spec (FRONTEND_BACKEND_CONTRACT.md):
  PREPROCESS_PLAN: {steps:[...]}
  SPLIT_SUMMARY: {train_rows, val_rows, test_rows, stratified, seed}

Handoff: UI displays preprocessing steps clearly.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler


class PreprocessStep(BaseModel):
    """A single preprocessing step in the plan."""
    
    step_type: str = Field(..., description="Type of step: impute, scale, encode, drop")
    columns: List[str] = Field(..., description="Columns affected by this step")
    method: str = Field(..., description="Method used (e.g., 'mean', 'standard', 'onehot')")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for this step"
    )
    reason: str = Field(..., description="Why this step is being applied")


class PreprocessPlan(BaseModel):
    """Contract payload for PREPROCESS_PLAN event."""
    
    steps: List[PreprocessStep] = Field(
        default_factory=list,
        description="Ordered list of preprocessing steps"
    )


class SplitSummary(BaseModel):
    """Contract payload for SPLIT_SUMMARY event."""
    
    train_rows: int = Field(..., description="Number of training rows")
    val_rows: int = Field(..., description="Number of validation rows")
    test_rows: int = Field(..., description="Number of test rows")
    stratified: bool = Field(..., description="Whether stratification was used")
    seed: int = Field(..., description="Random seed for reproducibility")


class PreprocessAgent:
    """
    Creates preprocessing pipeline plans for tabular ML.
    
    Analyzes dataset column types and creates a ColumnTransformer plan:
    - Numeric columns: impute (mean/median) + scale (standard/robust)
    - Categorical columns: impute (most_frequent) + encode (onehot)
    - Drop: ID columns, constant columns
    
    Usage:
        agent = PreprocessAgent()
        
        # Create preprocessing plan
        plan = agent.create_plan(
            df=dataframe,
            target_column="churn",
            task_type="classification"
        )
        # Returns: {steps: [{step_type, columns, method, params, reason}, ...]}
        
        # Get train/test split summary
        split_summary = agent.create_split_summary(
            df=dataframe,
            target_column="churn",
            task_type="classification",
            test_size=0.2,
            seed=42
        )
        # Returns: {train_rows, val_rows, test_rows, stratified, seed}
        
        # Build actual sklearn transformer (for training stage)
        transformer = agent.build_transformer(df, target_column="churn")
    """
    
    def __init__(
        self,
        numeric_impute_strategy: str = "mean",
        numeric_scaler: str = "standard",
        categorical_impute_strategy: str = "most_frequent",
        categorical_encoder: str = "onehot",
        drop_high_cardinality_threshold: float = 0.95,
    ):
        """
        Initialize preprocessing agent.
        
        Args:
            numeric_impute_strategy: How to fill missing numeric values (mean/median)
            numeric_scaler: How to scale numeric features (standard/robust)
            categorical_impute_strategy: How to fill missing categorical (most_frequent/constant)
            categorical_encoder: How to encode categorical (onehot)
            drop_high_cardinality_threshold: Drop categorical if unique_ratio > this (default 0.95)
        """
        self.numeric_impute_strategy = numeric_impute_strategy
        self.numeric_scaler = numeric_scaler
        self.categorical_impute_strategy = categorical_impute_strategy
        self.categorical_encoder = categorical_encoder
        self.drop_high_cardinality_threshold = drop_high_cardinality_threshold
    
    def create_plan(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create preprocessing plan (PREPROCESS_PLAN event payload).
        
        Args:
            df: Input dataframe
            target_column: Name of target column (will be excluded from features)
            task_type: ML task type (classification/regression)
            
        Returns:
            PREPROCESS_PLAN payload: {steps: [...]}
        """
        steps = []
        
        # Separate features from target
        feature_cols = [col for col in df.columns if col != target_column]
        
        # Classify columns
        numeric_cols, categorical_cols, cols_to_drop = self._classify_columns(df[feature_cols])
        
        # Step 1: Drop ID/constant columns
        if cols_to_drop:
            steps.append(PreprocessStep(
                step_type="drop",
                columns=cols_to_drop,
                method="remove",
                params={},
                reason="High cardinality (likely ID) or constant columns"
            ))
        
        # Step 2: Numeric imputation
        numeric_missing = [col for col in numeric_cols if df[col].isnull().sum() > 0]
        if numeric_missing:
            steps.append(PreprocessStep(
                step_type="impute",
                columns=numeric_missing,
                method=self.numeric_impute_strategy,
                params={"strategy": self.numeric_impute_strategy},
                reason=f"Fill missing numeric values with {self.numeric_impute_strategy}"
            ))
        
        # Step 3: Numeric scaling
        if numeric_cols:
            scaler_name = "StandardScaler" if self.numeric_scaler == "standard" else "RobustScaler"
            steps.append(PreprocessStep(
                step_type="scale",
                columns=numeric_cols,
                method=self.numeric_scaler,
                params={"scaler": scaler_name},
                reason=f"Normalize numeric features using {scaler_name}"
            ))
        
        # Step 4: Categorical imputation
        categorical_missing = [col for col in categorical_cols if df[col].isnull().sum() > 0]
        if categorical_missing:
            steps.append(PreprocessStep(
                step_type="impute",
                columns=categorical_missing,
                method=self.categorical_impute_strategy,
                params={"strategy": self.categorical_impute_strategy},
                reason=f"Fill missing categorical values with {self.categorical_impute_strategy}"
            ))
        
        # Step 5: Categorical encoding
        if categorical_cols:
            steps.append(PreprocessStep(
                step_type="encode",
                columns=categorical_cols,
                method=self.categorical_encoder,
                params={"encoder": "OneHotEncoder", "handle_unknown": "ignore"},
                reason="Convert categorical variables to numeric (one-hot encoding)"
            ))
        
        plan = PreprocessPlan(steps=steps)
        return plan.model_dump()
    
    def create_split_summary(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Create train/val/test split summary (SPLIT_SUMMARY event payload).
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            task_type: ML task type (determines stratification)
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining after test)
            seed: Random seed
            
        Returns:
            SPLIT_SUMMARY payload: {train_rows, val_rows, test_rows, stratified, seed}
        """
        # Drop rows where target is missing
        if target_column and target_column in df.columns:
            df_clean = df.dropna(subset=[target_column])
        else:
            df_clean = df
        
        n_total = len(df_clean)
        n_test = int(n_total * test_size)
        n_val = int((n_total - n_test) * val_size)
        n_train = n_total - n_test - n_val
        
        # Determine if stratification should be used
        stratified = False
        if task_type == "classification" and target_column:
            # Check if stratification is feasible (enough samples per class)
            if target_column in df_clean.columns:
                min_class_count = df_clean[target_column].value_counts().min()
                if min_class_count >= 2:  # Need at least 2 samples per class
                    stratified = True
        
        summary = SplitSummary(
            train_rows=n_train,
            val_rows=n_val,
            test_rows=n_test,
            stratified=stratified,
            seed=seed,
        )
        
        return summary.model_dump()
    
    def build_transformer(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> ColumnTransformer:
        """
        Build actual sklearn ColumnTransformer for training stage.
        
        Args:
            df: Input dataframe
            target_column: Name of target column (excluded from features)
            
        Returns:
            sklearn ColumnTransformer ready for Pipeline
        """
        # Separate features
        feature_cols = [col for col in df.columns if col != target_column]
        
        # Classify columns
        numeric_cols, categorical_cols, cols_to_drop = self._classify_columns(df[feature_cols])
        
        transformers = []
        
        # Numeric pipeline
        if numeric_cols:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.numeric_impute_strategy)),
                ('scaler', StandardScaler() if self.numeric_scaler == "standard" else RobustScaler()),
            ])
            transformers.append(('numeric', numeric_pipeline, numeric_cols))
        
        # Categorical pipeline
        if categorical_cols:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.categorical_impute_strategy)),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ])
            transformers.append(('categorical', categorical_pipeline, categorical_cols))
        
        # Create ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop ID/constant columns
            verbose_feature_names_out=False
        )
        
        return preprocessor
    
    def _classify_columns(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Classify columns as numeric, categorical, or to drop.
        
        Args:
            df: Dataframe to analyze
            
        Returns:
            (numeric_cols, categorical_cols, cols_to_drop)
        """
        numeric_cols = []
        categorical_cols = []
        cols_to_drop = []
        
        n_rows = len(df)
        
        for col in df.columns:
            # Skip if all missing
            if df[col].isnull().all():
                cols_to_drop.append(col)
                continue
            
            # Check if constant (no variance)
            if df[col].nunique() == 1:
                cols_to_drop.append(col)
                continue
            
            dtype = df[col].dtype
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_cols.append(col)
            
            # Categorical columns
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                # Check for high cardinality (likely ID column)
                n_unique = df[col].nunique()
                unique_ratio = n_unique / n_rows if n_rows > 0 else 0
                
                if unique_ratio > self.drop_high_cardinality_threshold:
                    cols_to_drop.append(col)
                else:
                    categorical_cols.append(col)
            
            # Other types (datetime, etc.) - drop for now
            else:
                cols_to_drop.append(col)
        
        return numeric_cols, categorical_cols, cols_to_drop


# Legacy compatibility
class PreprocessAgent_Legacy:
    """Legacy stub for backward compatibility."""
    
    def __init__(self):
        pass

    def handle_missing_values(self, data):
        # Use PreprocessAgent instead
        pass

    def encode_categorical_variables(self, data):
        # Use PreprocessAgent instead
        pass

    def scale_numerical_features(self, data):
        # Use PreprocessAgent instead
        pass

    def preprocess(self, data):
        # Use PreprocessAgent instead
        pass