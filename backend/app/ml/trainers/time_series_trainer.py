"""
Simple time-series trainer using lag features and RandomForestRegressor.
Intended for small demos; not production-grade forecasting.
"""
import asyncio
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus
from app.ml.artifacts import save_json_asset, asset_url


@dataclass
class TimeSeriesConfig:
    project_id: str
    value_col: str = "value"
    lags: int = 3
    test_size: float = 0.2
    steps: int = 30


class TimeSeriesTrainer:
    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.run_id = f"run_{uuid.uuid4().hex[:8]}"

    async def _emit(self, event_name: EventType, payload: Dict[str, Any], stage_status: StageStatus = StageStatus.IN_PROGRESS):
        await event_bus.publish_event(
            project_id=self.config.project_id,
            event_name=event_name,
            payload=payload,
            stage_id=StageID.TRAIN,
            stage_status=stage_status,
        )

    def _build_lag_df(self, df: pd.DataFrame) -> pd.DataFrame:
        series = df[self.config.value_col].reset_index(drop=True)
        lagged = pd.concat([series.shift(i) for i in range(self.config.lags + 1)], axis=1)
        lagged.columns = [f"lag_{i}" for i in range(self.config.lags + 1)]
        lagged = lagged.dropna().reset_index(drop=True)
        X = lagged[[c for c in lagged.columns if c != "lag_0"]]
        y = lagged["lag_0"]
        return X, y

    async def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.config.value_col not in df.columns:
            raise ValueError(f"value_col '{self.config.value_col}' not found")
        X, y = self._build_lag_df(df)
        if len(X) < 5:
            raise ValueError("Not enough data for time-series training")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42)

        await self._emit(
            EventType.TRAIN_RUN_STARTED,
            {
                "run_id": self.run_id,
                "model_id": "rf_ts",
                "metric_primary": "rmse",
                "config": {"task_type": "time_series", "lags": self.config.lags},
            },
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)

        residuals = (y_test - preds).tolist()
        resid_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_residuals.json", {"residuals": residuals})

        await self._emit(EventType.RESIDUALS_PLOT_READY, {"asset_url": asset_url(resid_path)})
        await self._emit(
            EventType.TRAIN_RUN_FINISHED,
            {"run_id": self.run_id, "status": "success", "final_metrics": {"rmse": float(rmse)}},
            stage_status=StageStatus.COMPLETED,
        )
        return {"metrics": {"rmse": float(rmse)}, "run_id": self.run_id, "artifacts": {"residuals": asset_url(resid_path)}}
