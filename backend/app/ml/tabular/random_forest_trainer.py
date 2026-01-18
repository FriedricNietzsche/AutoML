"""
Random Forest trainer for tabular data
Uses sklearn's RandomForest - excellent for tabular classification/regression
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import joblib


class RandomForestTrainer:
    """
    Random Forest trainer for tabular data
    RandomForest is one of the best models for tabular data:
    - Handles non-linear relationships
    - Feature importance built-in
    - Robust to outliers
    - No feature scaling needed
    """
    
    def __init__(self, task_type: Literal["classification", "regression"] = "classification"):
        self.task_type = task_type
        self.model = None
        self.feature_names = None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
            
        Returns:
            Dictionary with training metrics
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
        
        print(f"[RandomForest] Training {self.task_type} model...")
        print(f"[RandomForest] Training samples: {len(X_train)}")
        print(f"[RandomForest] Features: {X_train.shape[1]}")
        
        self.feature_names = list(X_train.columns)
        
        # Choose model based on task type
        if self.task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,  # Use all CPU cores
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
            )
        
        # Train
        print(f"[RandomForest] Fitting {n_estimators} trees...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_preds = self.model.predict(X_train)
        
        metrics = {
            "num_samples": len(X_train),
            "num_features": X_train.shape[1],
            "n_estimators": n_estimators,
        }
        
        if self.task_type == "classification":
            train_acc = accuracy_score(y_train, train_preds)
            train_f1 = f1_score(y_train, train_preds, average='weighted')
            metrics["train_accuracy"] = train_acc
            metrics["train_f1"] = train_f1
            print(f"[RandomForest] Train accuracy: {train_acc:.4f}")
        else:
            train_r2 = r2_score(y_train, train_preds)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            metrics["train_r2"] = train_r2
            metrics["train_rmse"] = train_rmse
            print(f"[RandomForest] Train R²: {train_r2:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            
            if self.task_type == "classification":
                val_acc = accuracy_score(y_val, val_preds)
                val_f1 = f1_score(y_val, val_preds, average='weighted')
                metrics["val_accuracy"] = val_acc
                metrics["val_f1"] = val_f1
                print(f"[RandomForest] Val accuracy: {val_acc:.4f}")
            else:
                val_r2 = r2_score(y_val, val_preds)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                metrics["val_r2"] = val_r2
                metrics["val_rmse"] = val_rmse
                print(f"[RandomForest] Val R²: {val_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"[RandomForest] Top 5 important features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        metrics["feature_importance"] = feature_importance.to_dict('records')
        
        print(f"[RandomForest] ✅ Training complete!")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features match
        if set(X.columns) != set(self.feature_names):
            raise ValueError(f"Features don't match. Expected: {self.feature_names}")
        
        X = X[self.feature_names]  # Reorder columns
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)"""
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = X[self.feature_names]
        return self.model.predict_proba(X)
    
    def save(self, save_path: Path):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
        }, save_path)
        print(f"[RandomForest] ✅ Model saved to {save_path}")
    
    def load(self, save_path: Path):
        """Load model"""
        data = joblib.load(save_path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.task_type = data['task_type']
        print(f"[RandomForest] ✅ Model loaded from {save_path}")
