"""
Tabular training runner with streaming events.
- Uses sklearn pipelines
- Emits TRAIN_* events per contract
"""
import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os

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

        y_pred = pipeline.predict(X_test)
        metrics: Dict[str, float] = {}
        artifacts: Dict[str, str] = {}
        if is_classification:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            metrics = {"accuracy": float(acc), "f1": float(f1)}
            cm = confusion_matrix(y_test, y_pred).tolist()
            cm_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_confusion.json", {"confusion": cm})
            artifacts["confusion"] = asset_url(cm_path)
            await self._emit(
                EventType.METRIC_SCALAR,
                {"run_id": self.run_id, "name": "accuracy", "split": "test", "step": self.config.steps, "value": float(acc)},
            )
            await self._emit(
                EventType.CONFUSION_MATRIX_READY,
                {"asset_url": artifacts["confusion"]},
            )
            await self._emit(
                EventType.ARTIFACT_ADDED,
                {"artifact": {"id": f"{self.run_id}_confusion", "type": "confusion_matrix", "name": "Confusion Matrix", "url": artifacts["confusion"], "meta": {"kind": "confusion_matrix", "matrix": cm}}},
            )
            # Feature importance
            try:
                model = pipeline.named_steps["model"]
                importances = model.feature_importances_
                feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
                ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                fi_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_feature_importance.json", {"feature_importance": ranked})
                artifacts["feature_importance"] = asset_url(fi_path)
                await self._emit(
                    EventType.FEATURE_IMPORTANCE_READY,
                    {"asset_url": artifacts["feature_importance"]},
                )
                await self._emit(
                    EventType.ARTIFACT_ADDED,
                    {"artifact": {"id": f"{self.run_id}_fi", "type": "feature_importance", "name": "Feature Importance", "url": artifacts["feature_importance"], "meta": {"kind": "feature_importance", "ranking": ranked}}},
                )
            except Exception:
                pass
        else:
            # Some sklearn versions do not support the `squared` kwarg, so compute RMSE manually.
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(mse**0.5)
            r2 = r2_score(y_test, y_pred)
            metrics = {"rmse": float(rmse), "r2": float(r2)}
            await self._emit(
                EventType.METRIC_SCALAR,
                {"run_id": self.run_id, "name": "rmse", "split": "test", "step": self.config.steps, "value": float(rmse)},
            )
            # Residuals
            residuals = (y_test - y_pred).tolist()
            resid_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_residuals.json", {"residuals": residuals})
            artifacts["residuals"] = asset_url(resid_path)
            await self._emit(
                EventType.RESIDUALS_PLOT_READY,
                {"asset_url": artifacts["residuals"]},
            )
            await self._emit(
                EventType.ARTIFACT_ADDED,
                {"artifact": {"id": f"{self.run_id}_resid", "type": "residuals", "name": "Residuals", "url": artifacts["residuals"], "meta": {"kind": "residuals", "points": residuals}}},
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

        return {"metrics": metrics, "run_id": self.run_id, "artifacts": artifacts}
