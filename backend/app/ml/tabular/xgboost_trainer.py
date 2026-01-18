"""
XGBoost trainer for tabular data
XGBoost is state-of-the-art for tabular data competitions (Kaggle winner)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import joblib


class XGBoostTrainer:
    """
    XGBoost trainer for tabular data
    XGBoost is the gold standard for tabular ML:
    - Dominates Kaggle competitions
    - Handles missing values
    - Built-in regularization
    - Fast training with GPU support
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
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Row sampling ratio
            colsample_bytree: Column sampling ratio
            random_state: Random seed
            
        Returns:
            Dictionary with training metrics
        """
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
            
            print(f"[XGBoost] Training {self.task_type} model...")
            print(f"[XGBoost] Training samples: {len(X_train)}")
            print(f"[XGBoost] Features: {X_train.shape[1]}")
            
            self.feature_names = list(X_train.columns)
            
            # Choose objective based on task type
            if self.task_type == "classification":
                # Check if binary or multiclass
                n_classes = len(np.unique(y_train))
                if n_classes == 2:
                    objective = "binary:logistic"
                    eval_metric = "logloss"
                else:
                    objective = "multi:softmax"
                    eval_metric = "mlogloss"
                    
                self.model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=random_state,
                    objective=objective,
                    eval_metric=eval_metric,
                    tree_method='hist',  # Fast histogram-based algorithm
                    n_jobs=-1,
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=random_state,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    tree_method='hist',
                    n_jobs=-1,
                )
            
            # Prepare evaluation set
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            # Train with early stopping
            print(f"[XGBoost] Training with early stopping...")
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
            )
            
            # Evaluate on training set
            train_preds = self.model.predict(X_train)
            
            metrics = {
                "num_samples": len(X_train),
                "num_features": X_train.shape[1],
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
            }
            
            if self.task_type == "classification":
                train_acc = accuracy_score(y_train, train_preds)
                train_f1 = f1_score(y_train, train_preds, average='weighted')
                metrics["train_accuracy"] = train_acc
                metrics["train_f1"] = train_f1
                print(f"[XGBoost] Train accuracy: {train_acc:.4f}")
            else:
                train_r2 = r2_score(y_train, train_preds)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
                metrics["train_r2"] = train_r2
                metrics["train_rmse"] = train_rmse
                print(f"[XGBoost] Train R²: {train_r2:.4f}")
            
            # Evaluate on validation set
            if X_val is not None and y_val is not None:
                val_preds = self.model.predict(X_val)
                
                if self.task_type == "classification":
                    val_acc = accuracy_score(y_val, val_preds)
                    val_f1 = f1_score(y_val, val_preds, average='weighted')
                    metrics["val_accuracy"] = val_acc
                    metrics["val_f1"] = val_f1
                    print(f"[XGBoost] Val accuracy: {val_acc:.4f}")
                else:
                    val_r2 = r2_score(y_val, val_preds)
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                    metrics["val_r2"] = val_r2
                    metrics["val_rmse"] = val_rmse
                    print(f"[XGBoost] Val R²: {val_r2:.4f}")
            
            # Feature importance
            importance_dict = self.model.get_booster().get_score(importance_type='gain')
            feature_importance = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in importance_dict.items()
            ]).sort_values('importance', ascending=False)
            
            print(f"[XGBoost] Top 5 important features:")
            for idx, row in feature_importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            metrics["feature_importance"] = feature_importance.to_dict('records')
            
            print(f"[XGBoost] ✅ Training complete!")
            return metrics
            
        except ImportError:
            print(f"[XGBoost] ⚠️  XGBoost not installed. Install with: pip install xgboost")
            raise ImportError("XGBoost library not available. Please install xgboost.")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features match
        if set(X.columns) != set(self.feature_names):
            raise ValueError(f"Features don't match. Expected: {self.feature_names}")
        
        X = X[self.feature_names]
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
        print(f"[XGBoost] ✅ Model saved to {save_path}")
    
    def load(self, save_path: Path):
        """Load model"""
        data = joblib.load(save_path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.task_type = data['task_type']
        print(f"[XGBoost] ✅ Model loaded from {save_path}")
