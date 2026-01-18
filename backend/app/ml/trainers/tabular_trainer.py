"""
Tabular training runner with streaming events.
- Uses sklearn pipelines
- Emits TRAIN_* events per contract
- Computes comprehensive evaluation metrics
"""
import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus
from app.api.assets import ASSET_ROOT
from app.ml.artifacts import save_json_asset, asset_url
from app.ml.metrics import MetricsCalculator, SHAPExplainer, SHAP_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    project_id: str
    target: str
    task_type: str  # classification|regression
    model_id: str = "auto"  # auto|rf|xgb|logreg|linear
    steps: int = 50
    test_size: float = 0.2
    random_state: int = 42


class TabularTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.run_id = f"run_{uuid.uuid4().hex[:8]}"
        if os.getenv("FAST_TEST"):
            self.config.steps = min(self.config.steps, 10)

    def _build_pipeline(self, df: pd.DataFrame) -> Pipeline:
        target = self.config.target
        X = df.drop(columns=[target])
        num_cols = X.select_dtypes(include=["int", "float"]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=["int", "float"]).columns.tolist()

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )
        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

        model_id = self.config.model_id
        fast = bool(os.getenv("FAST_TEST"))
        if self.config.task_type == "classification":
            if model_id == "xgb" and XGBClassifier is not None:
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    eval_metric="logloss",
                )
            elif model_id == "logreg":
                model = LogisticRegression(max_iter=150 if fast else 300)
            else:
                n_estimators = 10 if fast else 50
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=self.config.random_state)
        else:
            if model_id == "xgb" and XGBRegressor is not None:
                model = XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config.random_state,
                    objective="reg:squarederror",
                )
            elif model_id == "linear":
                model = LinearRegression()
            else:
                n_estimators = 10 if fast else 50
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=self.config.random_state)

        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        return pipeline

    async def _emit(self, event_name: EventType, payload: Dict[str, Any], stage_status: StageStatus = StageStatus.IN_PROGRESS):
        await event_bus.publish_event(
            project_id=self.config.project_id,
            event_name=event_name,
            payload=payload,
            stage_id=StageID.TRAIN,
            stage_status=stage_status,
        )

    async def _stream_progress(self, total_steps: int):
        for step in range(total_steps):
            await self._emit(
                EventType.TRAIN_PROGRESS,
                {
                    "run_id": self.run_id,
                    "epoch": 1,
                    "epochs": 1,
                    "step": step + 1,
                    "steps": total_steps,
                    "eta_s": max(0, total_steps - step - 1) * 0.05,
                    "phase": "fit",
                },
            )
            await self._emit(
                EventType.METRIC_SCALAR,
                {
                    "run_id": self.run_id,
                    "name": "loss",
                    "split": "train",
                    "step": step + 1,
                    "value": float(np.exp(-step / total_steps) + np.random.rand() * 0.05),
                },
            )
            await asyncio.sleep(0.05)

    async def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        target = self.config.target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        df = df.dropna(subset=[target])
        X = df.drop(columns=[target])
        y = df[target]

        task_type = self.config.task_type
        is_classification = task_type == "classification"

        test_size = self.config.test_size
        stratify = y if is_classification else None

        # Handle small-sample / low-class-count cases to avoid stratify errors.
        if is_classification:
            class_counts = y.value_counts()
            if len(df) < 10 or class_counts.min() < 2:
                stratify = None
            # Ensure test_size is valid given n_samples
            max_test = max(1, int(len(df) * 0.2))
            if isinstance(test_size, float):
                test_size = min(test_size, (len(df) - 1) / len(df))
            else:
                test_size = min(test_size, len(df) - 1, max_test)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_state, stratify=stratify
        )

        pipeline = self._build_pipeline(df)

        await self._emit(
            EventType.TRAIN_RUN_STARTED,
            {
                "run_id": self.run_id,
                "model_id": "auto",
                "metric_primary": "accuracy" if is_classification else "rmse",
                "config": {
                    "task_type": task_type,
                    "target": target,
                    "split": {"test_size": self.config.test_size, "random_state": self.config.random_state},
                },
            },
        )

        # Stream synthetic progress while fitting
        progress_task = asyncio.create_task(self._stream_progress(self.config.steps))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, pipeline.fit, X_train, y_train)
        await progress_task

        # =====================================================================
        # EVALUATION PHASE - Comprehensive metrics with real-time streaming
        # =====================================================================
        await self._emit(
            EventType.EVALUATION_STARTED,
            {
                "run_id": self.run_id,
                "task_type": task_type,
                "message": "Computing comprehensive evaluation metrics...",
            },
        )

        y_pred = pipeline.predict(X_test)
        
        # Get probabilities for classification
        y_prob = None
        if is_classification and hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")

        metrics: Dict[str, float] = {}
        artifacts: Dict[str, str] = {}
        all_metrics: Dict[str, Any] = {}

        if is_classification:
            # Compute comprehensive classification metrics
            all_metrics = MetricsCalculator.classification_metrics(
                y_test, y_pred, y_prob, labels=None
            )
            
            # Basic metrics for backward compatibility
            metrics = {
                "accuracy": all_metrics.get("accuracy", 0.0),
                "f1": all_metrics.get("f1", 0.0),
            }
            
            # Stream accuracy metric
            await self._emit(
                EventType.METRIC_SCALAR,
                {"run_id": self.run_id, "name": "accuracy", "split": "test", "step": self.config.steps, "value": all_metrics["accuracy"]},
            )
            
            # Emit comprehensive classification metrics
            await self._emit(
                EventType.CLASSIFICATION_METRICS_READY,
                {
                    "run_id": self.run_id,
                    "accuracy": all_metrics.get("accuracy", 0.0),
                    "balanced_accuracy": all_metrics.get("balanced_accuracy", 0.0),
                    "precision": all_metrics.get("precision", 0.0),
                    "recall": all_metrics.get("recall", 0.0),
                    "f1": all_metrics.get("f1", 0.0),
                    "roc_auc": all_metrics.get("roc_auc"),
                    "mcc": all_metrics.get("mcc", 0.0),
                    "cohen_kappa": all_metrics.get("cohen_kappa", 0.0),
                    "log_loss": all_metrics.get("log_loss"),
                    "average_precision": all_metrics.get("average_precision"),
                    "specificity": all_metrics.get("specificity"),
                    "sensitivity": all_metrics.get("sensitivity"),
                    "n_classes": all_metrics.get("n_classes", 2),
                    "class_labels": all_metrics.get("class_labels", []),
                    "class_distribution": all_metrics.get("class_distribution", {}),
                    "precision_per_class": all_metrics.get("precision_per_class"),
                    "recall_per_class": all_metrics.get("recall_per_class"),
                    "f1_per_class": all_metrics.get("f1_per_class"),
                },
            )
            
            # Confusion matrix
            cm = all_metrics.get("confusion_matrix", [])
            cm_data = {
                "confusion": cm,
                "labels": all_metrics.get("class_labels", []),
                "accuracy": all_metrics.get("accuracy"),
                "precision": all_metrics.get("precision"),
                "recall": all_metrics.get("recall"),
                "f1": all_metrics.get("f1"),
            }
            cm_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_confusion.json", cm_data)
            artifacts["confusion"] = asset_url(cm_path)
            
            await self._emit(
                EventType.CONFUSION_MATRIX_READY,
                {
                    "run_id": self.run_id,
                    "matrix": cm,
                    "labels": all_metrics.get("class_labels"),
                    "true_positives": all_metrics.get("true_positives"),
                    "true_negatives": all_metrics.get("true_negatives"),
                    "false_positives": all_metrics.get("false_positives"),
                    "false_negatives": all_metrics.get("false_negatives"),
                    "asset_url": artifacts["confusion"],
                },
            )
            await self._emit(
                EventType.ARTIFACT_ADDED,
                {"artifact": {"id": f"{self.run_id}_confusion", "type": "confusion_matrix", "name": "Confusion Matrix", "url": artifacts["confusion"], "meta": {"accuracy": all_metrics.get("accuracy")}}},
            )
            
            # ROC Curve (if available)
            if "roc_curve" in all_metrics:
                roc_data = all_metrics["roc_curve"]
                roc_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_roc_curve.json", {
                    "fpr": roc_data["fpr"],
                    "tpr": roc_data["tpr"],
                    "thresholds": roc_data["thresholds"],
                    "auc": all_metrics.get("roc_auc"),
                })
                artifacts["roc_curve"] = asset_url(roc_path)
                
                await self._emit(
                    EventType.ROC_CURVE_READY,
                    {
                        "run_id": self.run_id,
                        "fpr": roc_data["fpr"],
                        "tpr": roc_data["tpr"],
                        "thresholds": roc_data["thresholds"],
                        "auc": all_metrics.get("roc_auc"),
                        "asset_url": artifacts["roc_curve"],
                    },
                )
                await self._emit(
                    EventType.ARTIFACT_ADDED,
                    {"artifact": {"id": f"{self.run_id}_roc", "type": "roc_curve", "name": "ROC Curve", "url": artifacts["roc_curve"], "meta": {"auc": all_metrics.get("roc_auc")}}},
                )
            
            # Precision-Recall Curve (if available)
            if "pr_curve" in all_metrics:
                pr_data = all_metrics["pr_curve"]
                pr_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_pr_curve.json", {
                    "precision": pr_data["precision"],
                    "recall": pr_data["recall"],
                    "thresholds": pr_data["thresholds"],
                    "average_precision": all_metrics.get("average_precision"),
                })
                artifacts["pr_curve"] = asset_url(pr_path)
                
                await self._emit(
                    EventType.PRECISION_RECALL_CURVE_READY,
                    {
                        "run_id": self.run_id,
                        "precision": pr_data["precision"],
                        "recall": pr_data["recall"],
                        "thresholds": pr_data["thresholds"],
                        "average_precision": all_metrics.get("average_precision"),
                        "asset_url": artifacts["pr_curve"],
                    },
                )
                await self._emit(
                    EventType.ARTIFACT_ADDED,
                    {"artifact": {"id": f"{self.run_id}_pr", "type": "pr_curve", "name": "Precision-Recall Curve", "url": artifacts["pr_curve"], "meta": {"average_precision": all_metrics.get("average_precision")}}},
                )

            # Feature importance
            try:
                model = pipeline.named_steps["model"]
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
                    ranked = [{"feature": str(f), "importance": float(i)} for f, i in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)]
                    fi_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_feature_importance.json", {"feature_importance": ranked})
                    artifacts["feature_importance"] = asset_url(fi_path)
                    
                    await self._emit(
                        EventType.FEATURE_IMPORTANCE_READY,
                        {
                            "run_id": self.run_id,
                            "features": ranked,
                            "method": "model",
                            "asset_url": artifacts["feature_importance"],
                        },
                    )
                    await self._emit(
                        EventType.ARTIFACT_ADDED,
                        {"artifact": {"id": f"{self.run_id}_fi", "type": "feature_importance", "name": "Feature Importance", "url": artifacts["feature_importance"], "meta": {}}},
                    )
            except Exception as e:
                logger.warning(f"Could not compute feature importance: {e}")
            
            # SHAP explanations (if available)
            if SHAP_AVAILABLE:
                try:
                    model = pipeline.named_steps["model"]
                    # Transform training data for SHAP
                    X_train_transformed = pipeline.named_steps["preprocess"].transform(X_train)
                    X_test_transformed = pipeline.named_steps["preprocess"].transform(X_test)
                    
                    explainer = SHAPExplainer(model)
                    explainer.fit(X_train_transformed, max_samples=50)
                    shap_results = explainer.explain(X_test_transformed, max_samples=50)
                    
                    if shap_results.get("available"):
                        shap_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_shap.json", shap_results)
                        artifacts["shap"] = asset_url(shap_path)
                        
                        await self._emit(
                            EventType.SHAP_EXPLANATIONS_READY,
                            {
                                "run_id": self.run_id,
                                "available": True,
                                "feature_names": shap_results.get("feature_names"),
                                "global_importance": shap_results.get("global_importance"),
                                "importance_ranking": shap_results.get("importance_ranking"),
                                "asset_url": artifacts["shap"],
                            },
                        )
                        await self._emit(
                            EventType.ARTIFACT_ADDED,
                            {"artifact": {"id": f"{self.run_id}_shap", "type": "shap_explanations", "name": "SHAP Explanations", "url": artifacts["shap"], "meta": {}}},
                        )
                except Exception as e:
                    logger.warning(f"Could not compute SHAP explanations: {e}")

        else:
            # REGRESSION METRICS
            all_metrics = MetricsCalculator.regression_metrics(y_test, y_pred)
            
            # Basic metrics for backward compatibility
            metrics = {
                "rmse": all_metrics.get("rmse", 0.0),
                "r2": all_metrics.get("r2", 0.0),
            }
            
            # Stream RMSE metric
            await self._emit(
                EventType.METRIC_SCALAR,
                {"run_id": self.run_id, "name": "rmse", "split": "test", "step": self.config.steps, "value": all_metrics["rmse"]},
            )
            
            # Emit comprehensive regression metrics
            await self._emit(
                EventType.REGRESSION_METRICS_READY,
                {
                    "run_id": self.run_id,
                    "mse": all_metrics.get("mse", 0.0),
                    "rmse": all_metrics.get("rmse", 0.0),
                    "mae": all_metrics.get("mae", 0.0),
                    "median_ae": all_metrics.get("median_ae", 0.0),
                    "r2": all_metrics.get("r2", 0.0),
                    "explained_variance": all_metrics.get("explained_variance", 0.0),
                    "max_error": all_metrics.get("max_error", 0.0),
                    "mape": all_metrics.get("mape"),
                    "smape": all_metrics.get("smape"),
                    "n_samples": all_metrics.get("n_samples", len(y_test)),
                },
            )
            
            # Residuals
            residuals_data = all_metrics.get("residuals", {})
            resid_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_residuals.json", {
                "residuals": residuals_data.get("values", []),
                "mean": residuals_data.get("mean"),
                "std": residuals_data.get("std"),
                "y_true_stats": all_metrics.get("y_true_stats"),
                "y_pred_stats": all_metrics.get("y_pred_stats"),
            })
            artifacts["residuals"] = asset_url(resid_path)
            
            await self._emit(
                EventType.RESIDUALS_PLOT_READY,
                {"asset_url": artifacts["residuals"]},
            )
            await self._emit(
                EventType.ARTIFACT_ADDED,
                {"artifact": {"id": f"{self.run_id}_resid", "type": "residuals", "name": "Residuals", "url": artifacts["residuals"], "meta": {"rmse": all_metrics.get("rmse")}}},
            )
            
            # Feature importance for regression
            try:
                model = pipeline.named_steps["model"]
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
                    ranked = [{"feature": str(f), "importance": float(i)} for f, i in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)]
                    fi_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_feature_importance.json", {"feature_importance": ranked})
                    artifacts["feature_importance"] = asset_url(fi_path)
                    
                    await self._emit(
                        EventType.FEATURE_IMPORTANCE_READY,
                        {
                            "run_id": self.run_id,
                            "features": ranked,
                            "method": "model",
                            "asset_url": artifacts["feature_importance"],
                        },
                    )
                    await self._emit(
                        EventType.ARTIFACT_ADDED,
                        {"artifact": {"id": f"{self.run_id}_fi", "type": "feature_importance", "name": "Feature Importance", "url": artifacts["feature_importance"], "meta": {}}},
                    )
            except Exception as e:
                logger.warning(f"Could not compute feature importance: {e}")

        # Save comprehensive metrics as a single artifact
        full_metrics_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_full_metrics.json", {
            "task_type": task_type,
            "metrics": all_metrics,
            "run_id": self.run_id,
        })
        artifacts["full_metrics"] = asset_url(full_metrics_path)
        
        # Emit evaluation complete
        primary_metric = "accuracy" if is_classification else "rmse"
        primary_value = all_metrics.get(primary_metric, 0.0)
        
        await self._emit(
            EventType.EVALUATION_COMPLETE,
            {
                "run_id": self.run_id,
                "task_type": task_type,
                "primary_metric": primary_metric,
                "primary_value": primary_value,
                "all_metrics": all_metrics,
                "artifacts": [{"type": k, "url": v} for k, v in artifacts.items()],
                "shap_available": "shap" in artifacts,
            },
        )

        await self._emit(
            EventType.TRAIN_RUN_FINISHED,
            {
                "run_id": self.run_id,
                "status": "success",
                "final_metrics": metrics,
            },
            stage_status=StageStatus.COMPLETED,
        )

        return {"metrics": metrics, "all_metrics": all_metrics, "run_id": self.run_id, "artifacts": artifacts}
