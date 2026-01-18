"""
Text Sentiment Classification using Transformers
Uses HuggingFace transformers for state-of-the-art sentiment analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import json


class SentimentClassifier:
    """
    Sentiment classifier using pre-trained transformers (DistilBERT)
    This is the modern, production-grade approach for text classification
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.label_map = None
        
    def train(
        self,
        train_texts: list,
        train_labels: list,
        val_texts: Optional[list] = None,
        val_labels: Optional[list] = None,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> Dict[str, Any]:
        """
        Fine-tune a pre-trained transformer model for sentiment classification
        
        Args:
            train_texts: List of training text samples
            train_labels: List of training labels (0 or 1)
            val_texts: Optional validation texts
            val_labels: Optional validation labels
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training metrics
        """
        print(f"[SentimentClassifier] Training transformer model: {self.model_name}")
        print(f"[SentimentClassifier] Training samples: {len(train_texts)}")
        
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
                DataCollatorWithPadding,
            )
            from datasets import Dataset
            import torch
            
            # Check if GPU is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[SentimentClassifier] Using device: {device}")
            
            # Load tokenizer and model
            print(f"[SentimentClassifier] Loading pre-trained model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Determine number of labels
            unique_labels = sorted(set(train_labels))
            num_labels = len(unique_labels)
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            print(f"[SentimentClassifier] Number of classes: {num_labels}")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
            
            # Prepare datasets
            print(f"[SentimentClassifier] Tokenizing texts...")
            train_dataset = Dataset.from_dict({
                "text": train_texts,
                "label": [self.label_map[label] for label in train_labels]
            })
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            
            # Prepare validation dataset if provided
            eval_dataset = None
            if val_texts and val_labels:
                eval_dataset = Dataset.from_dict({
                    "text": val_texts,
                    "label": [self.label_map[label] for label in val_labels]
                })
                eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_steps=10,
                evaluation_strategy="epoch" if eval_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="accuracy" if eval_dataset else None,
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # Compute metrics function
            def compute_metrics(eval_pred):
                from sklearn.metrics import accuracy_score, f1_score
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                accuracy = accuracy_score(labels, predictions)
                f1 = f1_score(labels, predictions, average='weighted')
                return {"accuracy": accuracy, "f1": f1}
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics if eval_dataset else None,
            )
            
            # Train the model
            print(f"[SentimentClassifier] Starting training ({num_epochs} epochs)...")
            train_result = trainer.train()
            
            # Evaluate
            metrics = {}
            if eval_dataset:
                print(f"[SentimentClassifier] Evaluating on validation set...")
                eval_result = trainer.evaluate()
                metrics["val_accuracy"] = eval_result.get("eval_accuracy", 0.0)
                metrics["val_f1"] = eval_result.get("eval_f1", 0.0)
            
            # Training metrics
            metrics["train_loss"] = train_result.training_loss
            metrics["num_epochs"] = num_epochs
            metrics["num_samples"] = len(train_texts)
            
            print(f"[SentimentClassifier] ✅ Training complete!")
            print(f"[SentimentClassifier] Metrics: {metrics}")
            
            return metrics
            
        except ImportError as e:
            print(f"[SentimentClassifier] ⚠️  Transformers not available: {e}")
            print(f"[SentimentClassifier] Falling back to simple TF-IDF + Logistic Regression")
            return self._train_simple_model(train_texts, train_labels, val_texts, val_labels)
    
    def _train_simple_model(
        self,
        train_texts: list,
        train_labels: list,
        val_texts: Optional[list] = None,
        val_labels: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Fallback: Simple TF-IDF + Logistic Regression (sklearn)
        Used when transformers library is not available
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        
        print(f"[SentimentClassifier] Using TF-IDF + Logistic Regression")
        
        # TF-IDF vectorization
        self.tokenizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X_train = self.tokenizer.fit_transform(train_texts)
        
        # Train logistic regression
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train, train_labels)
        
        # Evaluate
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        
        metrics = {
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "num_samples": len(train_texts),
            "model_type": "TF-IDF + LogisticRegression"
        }
        
        if val_texts and val_labels:
            X_val = self.tokenizer.transform(val_texts)
            val_preds = self.model.predict(X_val)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            metrics["val_accuracy"] = val_acc
            metrics["val_f1"] = val_f1
        
        print(f"[SentimentClassifier] ✅ Training complete: {metrics}")
        return metrics
    
    def predict(self, texts: list) -> np.ndarray:
        """
        Predict sentiment for a list of texts
        
        Args:
            texts: List of text samples
            
        Returns:
            Array of predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Check if using transformer or sklearn
        if hasattr(self.model, 'predict_proba'):
            # sklearn model
            X = self.tokenizer.transform(texts)
            return self.model.predict(X)
        else:
            # Transformer model
            from transformers import pipeline
            classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
            results = classifier(texts)
            return np.array([int(r['label'].split('_')[-1]) for r in results])
    
    def save(self, save_dir: Path):
        """Save model and tokenizer"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save_pretrained'):
            # Transformer model
            self.model.save_pretrained(save_dir / "model")
            self.tokenizer.save_pretrained(save_dir / "tokenizer")
        else:
            # sklearn model
            joblib.dump(self.model, save_dir / "model.joblib")
            joblib.dump(self.tokenizer, save_dir / "tokenizer.joblib")
        
        # Save label map
        with open(save_dir / "label_map.json", 'w') as f:
            json.dump(self.label_map, f)
        
        print(f"[SentimentClassifier] ✅ Model saved to {save_dir}")
    
    def load(self, save_dir: Path):
        """Load model and tokenizer"""
        save_dir = Path(save_dir)
        
        # Load label map
        with open(save_dir / "label_map.json", 'r') as f:
            self.label_map = json.load(f)
        
        # Try loading transformer model first
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(save_dir / "model")
            self.tokenizer = AutoTokenizer.from_pretrained(save_dir / "tokenizer")
            print(f"[SentimentClassifier] ✅ Loaded transformer model from {save_dir}")
        except:
            # Fall back to sklearn model
            self.model = joblib.load(save_dir / "model.joblib")
            self.tokenizer = joblib.load(save_dir / "tokenizer.joblib")
            print(f"[SentimentClassifier] ✅ Loaded sklearn model from {save_dir}")
