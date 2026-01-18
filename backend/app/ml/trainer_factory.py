"""
Trainer Factory - Selects the best model trainer for each task type
Uses expert-recommended libraries and models for each ML domain
"""
from typing import Dict, Any, Literal
from pathlib import Path
import pandas as pd


class TrainerFactory:
    """
    Factory for creating appropriate trainers based on task type and data characteristics
    
    Philosophy:
    - Text classification → Transformers (HuggingFace) or fallback to TF-IDF + sklearn
    - Tabular classification/regression → XGBoost (Kaggle gold standard) or RandomForest
    - Image classification → Transfer Learning (ResNet, EfficientNet, ViT)
    - Each task gets the industry-standard best-practice model
    """
    
    # Map model IDs from ModelSelector to trainer names
    MODEL_ID_MAP = {
        # Classification
        "logreg": "logistic_regression",
        "xgb_clf": "xgboost",
        "rf_clf": "random_forest",
        "gb_clf": "gradient_boosting",
        # Regression
        "linreg": "linear_regression",
        "xgb_reg": "xgboost",
        "rf_reg": "random_forest",
        "gb_reg": "gradient_boosting",
    }
    
    @staticmethod
    def get_trainer(
        task_type: Literal["text_classification", "tabular_classification", "tabular_regression", "image_classification"],
        model_name: str = "auto",
        num_classes: int = 2,
    ):
        """
        Get the appropriate trainer for a task
        
        Args:
            task_type: Type of ML task
            model_name: Specific model to use, or "auto" for best default
            
        Returns:
            Trainer instance
        """
        
        # Map model ID to trainer name if needed
        if model_name in TrainerFactory.MODEL_ID_MAP:
            original_model_name = model_name
            model_name = TrainerFactory.MODEL_ID_MAP[model_name]
            print(f"[TrainerFactory] Mapped model ID '{original_model_name}' → '{model_name}'")
        
        if task_type == "text_classification":
            # Text classification → Use transformers (BERT, DistilBERT, etc.)
            from app.ml.text.sentiment_classifier import SentimentClassifier
            
            if model_name == "auto":
                # Default: DistilBERT (fast, accurate, production-ready)
                model_name = "distilbert-base-uncased"
            
            print(f"[TrainerFactory] Creating text classifier: {model_name}")
            return SentimentClassifier(model_name=model_name)
        
        elif task_type == "image_classification":
            # Image classification → Use transfer learning (ResNet, EfficientNet, ViT)
            from app.ml.vision.image_classifier import ImageClassifier
            
            if model_name == "auto":
                # Default: ResNet50 (reliable, well-tested, 25M params)
                model_name = "resnet50"
            
            print(f"[TrainerFactory] Creating image classifier: {model_name}")
            return ImageClassifier(model_name=model_name, num_classes=num_classes)
        
        elif task_type == "tabular_classification":
            # Tabular classification → XGBoost (best) or RandomForest (good alternative)
            if model_name == "auto" or model_name == "xgboost":
                try:
                    from app.ml.tabular.xgboost_trainer import XGBoostTrainer
                    print(f"[TrainerFactory] Creating XGBoost classifier (Kaggle gold standard)")
                    return XGBoostTrainer(task_type="classification")
                except ImportError:
                    print(f"[TrainerFactory] XGBoost not available, using RandomForest")
                    from app.ml.tabular.random_forest_trainer import RandomForestTrainer
                    return RandomForestTrainer(task_type="classification")
            elif model_name in ["random_forest", "gradient_boosting", "logistic_regression"]:
                # Map all sklearn models to appropriate trainers
                if model_name == "random_forest":
                    from app.ml.tabular.random_forest_trainer import RandomForestTrainer
                    print(f"[TrainerFactory] Creating RandomForest classifier")
                    return RandomForestTrainer(task_type="classification")
                elif model_name == "gradient_boosting":
                    # Use XGBoost as a substitute for GradientBoosting (same family, better performance)
                    try:
                        from app.ml.tabular.xgboost_trainer import XGBoostTrainer
                        print(f"[TrainerFactory] Using XGBoost for GradientBoosting (better performance)")
                        return XGBoostTrainer(task_type="classification")
                    except ImportError:
                        from app.ml.tabular.random_forest_trainer import RandomForestTrainer
                        print(f"[TrainerFactory] Using RandomForest as fallback for GradientBoosting")
                        return RandomForestTrainer(task_type="classification")
                elif model_name == "logistic_regression":
                    # Use RandomForest as a substitute (more powerful, works for same tasks)
                    from app.ml.tabular.random_forest_trainer import RandomForestTrainer
                    print(f"[TrainerFactory] Using RandomForest for LogisticRegression (more powerful)")
                    return RandomForestTrainer(task_type="classification")
            else:
                raise ValueError(f"Unknown tabular classification model: {model_name}")
        
        elif task_type == "tabular_regression":
            # Tabular regression → XGBoost (best) or RandomForest (good alternative)
            if model_name == "auto" or model_name == "xgboost":
                try:
                    from app.ml.tabular.xgboost_trainer import XGBoostTrainer
                    print(f"[TrainerFactory] Creating XGBoost regressor (Kaggle gold standard)")
                    return XGBoostTrainer(task_type="regression")
                except ImportError:
                    print(f"[TrainerFactory] XGBoost not available, using RandomForest")
                    from app.ml.tabular.random_forest_trainer import RandomForestTrainer
                    return RandomForestTrainer(task_type="regression")
            elif model_name in ["random_forest", "gradient_boosting", "linear_regression"]:
                # Map all sklearn models to appropriate trainers
                if model_name == "random_forest":
                    from app.ml.tabular.random_forest_trainer import RandomForestTrainer
                    print(f"[TrainerFactory] Creating RandomForest regressor")
                    return RandomForestTrainer(task_type="regression")
                elif model_name == "gradient_boosting":
                    # Use XGBoost as a substitute for GradientBoosting (same family, better performance)
                    try:
                        from app.ml.tabular.xgboost_trainer import XGBoostTrainer
                        print(f"[TrainerFactory] Using XGBoost for GradientBoosting (better performance)")
                        return XGBoostTrainer(task_type="regression")
                    except ImportError:
                        from app.ml.tabular.random_forest_trainer import RandomForestTrainer
                        print(f"[TrainerFactory] Using RandomForest as fallback for GradientBoosting")
                        return RandomForestTrainer(task_type="regression")
                elif model_name == "linear_regression":
                    # Use RandomForest as a substitute (more powerful, works for same tasks)
                    from app.ml.tabular.random_forest_trainer import RandomForestTrainer
                    print(f"[TrainerFactory] Using RandomForest for LinearRegression (more powerful)")
                    return RandomForestTrainer(task_type="regression")
            else:
                raise ValueError(f"Unknown tabular regression model: {model_name}")
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    @staticmethod
    def detect_task_type(df: pd.DataFrame, target_column: str) -> str:
        """
        Automatically detect task type from dataset
        
        Args:
            df: Input DataFrame
            target_column: Name of target/label column
            
        Returns:
            Task type string
        """
        # Check for image paths column
        for col in df.columns:
            if col == target_column:
                continue
            if df[col].dtype == 'object':
                sample_val = str(df[col].iloc[0])
                if any(ext in sample_val.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']):
                    print(f"[TrainerFactory] Detected image path column: '{col}' → image_classification")
                    return "image_classification"
        
        # Check if there's a text column (high cardinality, long strings)
        for col in df.columns:
            if col == target_column:
                continue
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                avg_length = df[col].astype(str).str.len().mean()
                if unique_ratio > 0.9 and avg_length > 50:
                    print(f"[TrainerFactory] Detected text column: '{col}' → text_classification")
                    return "text_classification"
        
        # Check target column type
        if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
            # Categorical target → classification
            print(f"[TrainerFactory] Detected categorical target → tabular_classification")
            return "tabular_classification"
        else:
            # Continuous target → regression
            print(f"[TrainerFactory] Detected continuous target → tabular_regression")
            return "tabular_regression"
    
    @staticmethod
    def train_model(
        task_type: str,
        df: pd.DataFrame,
        target_column: str,
        model_name: str = "auto",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        End-to-end training pipeline
        
        Args:
            task_type: Type of ML task
            df: Input DataFrame (already preprocessed)
            target_column: Name of target column
            model_name: Model to use
            test_size: Fraction of data for validation
            random_state: Random seed
            
        Returns:
            Dictionary with model and metrics
        """
        from sklearn.model_selection import train_test_split
        import joblib
        
        print(f"[TrainerFactory] Starting training pipeline...")
        print(f"[TrainerFactory] Task: {task_type}, Model: {model_name}")
        print(f"[TrainerFactory] Dataset: {df.shape}")
        
        # Split features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if task_type.endswith('classification') else None
        )
        
        print(f"[TrainerFactory] Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        # Get appropriate trainer
        trainer = TrainerFactory.get_trainer(task_type, model_name)
        
        # Train based on task type
        if task_type == "text_classification":
            # For text, we need to extract text column
            text_col = None
            for col in X.columns:
                if X[col].dtype == 'object':
                    unique_ratio = X[col].nunique() / len(X)
                    avg_length = X[col].astype(str).str.len().mean()
                    if unique_ratio > 0.9 and avg_length > 50:
                        text_col = col
                        break
            
            if text_col is None:
                raise ValueError("No text column found for text classification")
            
            train_texts = X_train[text_col].tolist()
            train_labels = y_train.tolist()
            val_texts = X_val[text_col].tolist()
            val_labels = y_val.tolist()
            
            metrics = trainer.train(
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
            )
        else:
            # Tabular tasks
            metrics = trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )
        
        print(f"[TrainerFactory] ✅ Training complete!")
        return {
            "trainer": trainer,
            "metrics": metrics,
            "task_type": task_type,
            "model_name": model_name,
        }
