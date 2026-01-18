"""
TrainerAgent - Trains machine learning models
Handles model initialization, training, and saving
"""
from typing import Dict, Any, Callable, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import asyncio


class TrainerAgent:
    def __init__(self):
        self.model = None
        self.training_history = []
        self.metrics = {}
        
    def initialize_model(self, model_id: str, task_type: str):
        """
        Initialize a model based on model_id and task_type
        
        Args:
            model_id: Model identifier (e.g., 'rf', 'lr', 'svm')
            task_type: 'classification' or 'regression'
        """
        print(f"[TrainerAgent] Initializing model: {model_id} for {task_type}")
        
        if task_type == "classification":
            if model_id in ["rf", "random_forest"]:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                print("[TrainerAgent] ✅ Initialized RandomForestClassifier")
            elif model_id in ["lr", "logistic"]:
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(random_state=42, max_iter=1000)
                print("[TrainerAgent] ✅ Initialized LogisticRegression")
            elif model_id == "svm":
                from sklearn.svm import SVC
                self.model = SVC(random_state=42)
                print("[TrainerAgent] ✅ Initialized SVC")
            else:
                # Default to Random Forest
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                print(f"[TrainerAgent] ⚠️ Unknown model '{model_id}', using RandomForestClassifier")
        
        else:  # regression
            if model_id in ["rf", "random_forest"]:
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                print("[TrainerAgent] ✅ Initialized RandomForestRegressor")
            elif model_id in ["lr", "linear"]:
                from sklearn.linear_model import LinearRegression
                self.model = LinearRegression()
                print("[TrainerAgent] ✅ Initialized LinearRegression")
            else:
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                print(f"[TrainerAgent] ⚠️ Unknown model '{model_id}', using RandomForestRegressor")

    async def train(self, data_path: str, target_column: str = None, 
                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train the model on the provided dataset
        
        Args:
            data_path: Path to CSV file
            target_column: Name of target column (if None, uses last column)
            progress_callback: Optional async function to report progress
            
        Returns:
            Dictionary with training metrics
        """
        print(f"[TrainerAgent] Loading data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"[TrainerAgent] Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Determine target column
        if target_column is None:
            target_column = df.columns[-1]
            print(f"[TrainerAgent] Auto-detected target column: '{target_column}'")
        
        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(f"[TrainerAgent] Features: {X.shape}, Target: {y.shape}")
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"[TrainerAgent] Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Report progress
        if progress_callback:
            await progress_callback(10, "Data loaded and split")
        
        # Train model
        print(f"[TrainerAgent] Training {self.model.__class__.__name__}...")
        if progress_callback:
            await progress_callback(30, "Training started...")
        
        self.model.fit(X_train, y_train)
        
        if progress_callback:
            await progress_callback(70, "Training complete, evaluating...")
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"[TrainerAgent] ✅ Training complete!")
        print(f"  - Train score: {train_score:.4f}")
        print(f"  - Test score: {test_score:.4f}")
        
        self.metrics = {
            "train_score": float(train_score),
            "test_score": float(test_score),
            "model_name": self.model.__class__.__name__,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": list(X.columns),
            "target": target_column,
        }
        
        if progress_callback:
            await progress_callback(100, "Training and evaluation complete!")
        
        return self.metrics

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("[TrainerAgent] Evaluating model...")
        
        predictions = self.model.predict(X_test)
        
        # Calculate metrics based on task type
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        try:
            # Try classification metrics
            accuracy = accuracy_score(y_test, predictions)
            metrics = {
                "accuracy": float(accuracy),
                "type": "classification"
            }
            print(f"[TrainerAgent] Classification accuracy: {accuracy:.4f}")
        except:
            # Fall back to regression metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            metrics = {
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2),
                "type": "regression"
            }
            print(f"[TrainerAgent] Regression R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        return metrics

    def save_model(self, filepath: str) -> str:
        """Save trained model to disk"""
        print(f"[TrainerAgent] Saving model to {filepath}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"[TrainerAgent] ✅ Model saved")
        return filepath

    def load_model(self, filepath: str):
        """Load model from disk"""
        print(f"[TrainerAgent] Loading model from {filepath}")
        self.model = joblib.load(filepath)
        print(f"[TrainerAgent] ✅ Model loaded: {self.model.__class__.__name__}")
