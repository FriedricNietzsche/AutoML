"""
PreprocessAgent - Profiles and preprocesses datasets
Handles data profiling, missing values, encoding, and scaling
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from app.ml.text.preprocessing import extract_text_features, detect_text_columns


class PreprocessAgent:
    def __init__(self):
        self.profile_data = None
        self.preprocessing_steps = []

    def profile_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Profile a dataset and return statistics
        
        Args:
            dataset_path: Path to CSV file
            
        Returns:
            Dictionary with profile information
        """
        print(f"[PreprocessAgent] Profiling dataset: {dataset_path}")
        
        try:
            # Load dataset
            df = pd.read_csv(dataset_path)
            print(f"[PreprocessAgent] Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Basic stats
            profile = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": {},
                "numeric_columns": [],
                "categorical_columns": [],
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            }
            
            # Analyze each column
            for col in df.columns:
                # Missing values
                missing = df[col].isnull().sum()
                if missing > 0:
                    profile["missing_values"][col] = {
                        "count": int(missing),
                        "percentage": float(missing / len(df) * 100)
                    }
                
                # Categorize column type
                if pd.api.types.is_numeric_dtype(df[col]):
                    profile["numeric_columns"].append(col)
                else:
                    profile["categorical_columns"].append(col)
            
            # Summary stats for numeric columns
            profile["numeric_stats"] = {}
            for col in profile["numeric_columns"]:
                profile["numeric_stats"][col] = {
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                }
            
            # Unique value counts for categorical
            profile["categorical_stats"] = {}
            for col in profile["categorical_columns"]:
                unique_count = df[col].nunique()
                profile["categorical_stats"][col] = {
                    "unique_values": int(unique_count),
                    "most_common": df[col].mode()[0] if len(df[col].mode()) > 0 else None
                }
            
            # Overall summary
            total_missing = sum(df.isnull().sum())
            profile["summary"] = {
                "total_missing_values": int(total_missing),
                "missing_percentage": float(total_missing / (len(df) * len(df.columns)) * 100),
                "numeric_column_count": len(profile["numeric_columns"]),
                "categorical_column_count": len(profile["categorical_columns"]),
            }
            
            print(f"[PreprocessAgent] ✅ Profile complete:")
            print(f"  - {profile['summary']['numeric_column_count']} numeric columns")
            print(f"  - {profile['summary']['categorical_column_count']} categorical columns")
            print(f"  - {profile['summary']['missing_percentage']:.1f}% missing values")
            
            self.profile_data = profile
            return profile
            
        except Exception as e:
            print(f"[PreprocessAgent] ❌ Error profiling dataset: {e}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by filling with median/mode"""
        print(f"[PreprocessAgent] Handling missing values...")
        steps = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Fill numeric with median
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    steps.append(f"Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
                else:
                    # Fill categorical with mode
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown"
                    df[col].fillna(mode_val, inplace=True)
                    steps.append(f"Filled {missing_count} missing values in '{col}' with mode ('{mode_val}')")
        
        self.preprocessing_steps.extend(steps)
        print(f"[PreprocessAgent] ✅ Handled missing values in {len(steps)} columns")
        return df

    def encode_categorical_variables(self, df: pd.DataFrame, max_categories: int = 10) -> pd.DataFrame:
        """Encode categorical variables using one-hot or label encoding"""
        print(f"[PreprocessAgent] Encoding categorical variables...")
        steps = []
        
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                unique_count = df[col].nunique()
                
                if unique_count <= max_categories:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(col, axis=1, inplace=True)
                    steps.append(f"One-hot encoded '{col}' ({unique_count} categories) → {len(dummies.columns)} features")
                else:
                    # Label encoding for high cardinality
                    df[col] = pd.Categorical(df[col]).codes
                    steps.append(f"Label encoded '{col}' ({unique_count} categories)")
        
        self.preprocessing_steps.extend(steps)
        print(f"[PreprocessAgent] ✅ Encoded {len(steps)} categorical columns")
        return df

    def scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features to [0, 1] range"""
        print(f"[PreprocessAgent] Scaling numerical features...")
        steps = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            
            if max_val > min_val:  # Avoid division by zero
                df[col] = (df[col] - min_val) / (max_val - min_val)
                steps.append(f"Scaled '{col}' to [0, 1] range")
        
        self.preprocessing_steps.extend(steps)
        print(f"[PreprocessAgent] ✅ Scaled {len(steps)} numeric columns")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full preprocessing pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print(f"[PreprocessAgent] Starting full preprocessing pipeline...")
        self.preprocessing_steps = []
        
        df = self.handle_missing_values(df)
        df = self.encode_categorical_variables(df)
        df = self.scale_numerical_features(df)
        
        print(f"[PreprocessAgent] ✅ Preprocessing complete - {len(self.preprocessing_steps)} steps applied")
        return df
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps applied"""
        return {
            "steps": self.preprocessing_steps,
            "step_count": len(self.preprocessing_steps),
            "profile": self.profile_data
        }
