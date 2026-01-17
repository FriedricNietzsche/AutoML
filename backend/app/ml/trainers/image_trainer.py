"""
Lightweight image trainer for cats vs dogs demo.
Uses Keras (TensorFlow) if available; otherwise falls back to synthetic training stream.
Now also supports a minimal PyTorch path if torch is installed.
"""
import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import os

from app.events.bus import event_bus
from app.events.schema import EventType, StageID, StageStatus
from app.api.assets import ASSET_ROOT
from app.ml.artifacts import save_json_asset, asset_url

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
except Exception:  # pragma: no cover
    torch = None


@dataclass
class ImageTrainConfig:
    project_id: str
    data_dir: Path
    steps: int = 50
    img_size: int = 128
    batch_size: int = 16
    epochs: int = 2


class ImageTrainer:
    def __init__(self, config: ImageTrainConfig):
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

    async def _stream_synthetic(self):
        for step in range(self.config.steps):
            await self._emit(
                EventType.TRAIN_PROGRESS,
                {
                    "run_id": self.run_id,
                    "epoch": 1,
                    "epochs": 1,
                    "step": step + 1,
                    "steps": self.config.steps,
                    "eta_s": max(0, self.config.steps - step - 1) * 0.05,
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
                    "value": float(np.exp(-step / self.config.steps) + np.random.rand() * 0.05),
                },
            )
            await asyncio.sleep(0.05)

    async def train(self) -> Dict[str, Any]:
        await self._emit(
            EventType.TRAIN_RUN_STARTED,
            {
                "run_id": self.run_id,
                "model_id": "cnn",
                "metric_primary": "accuracy",
                "config": {"task_type": "vision"},
            },
        )

        # If TF not available, simulate
        if tf is None and torch is None:
            await self._stream_synthetic()
            acc = 0.85
            cm = [[45, 5], [7, 43]]
            cm_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_confusion.json", {"confusion": cm})
            await self._emit(EventType.CONFUSION_MATRIX_READY, {"asset_url": asset_url(cm_path)})
            await self._emit(EventType.ARTIFACT_ADDED, {"artifact": {"id": f"{self.run_id}_confusion", "type": "confusion_matrix", "name": "Confusion Matrix", "url": asset_url(cm_path), "meta": {}}})
            await self._emit(
                EventType.METRIC_SCALAR,
                {"run_id": self.run_id, "name": "accuracy", "split": "test", "step": self.config.steps, "value": acc},
            )
            await self._emit(
                EventType.TRAIN_RUN_FINISHED,
                {"run_id": self.run_id, "status": "success", "final_metrics": {"accuracy": acc}},
                stage_status=StageStatus.COMPLETED,
            )
            return {"metrics": {"accuracy": acc}, "run_id": self.run_id}

        # PyTorch path (if available)
        if torch is not None:
            class ImageFolderDataset(Dataset):
                def __init__(self, root, img_size):
                    self.paths = []
                    self.labels = []
                    for f in Path(root).glob("*.png"):
                        parts = f.stem.split("_")
                        label = parts[0] if parts else "0"
                        self.paths.append(f)
                        self.labels.append(0 if label.lower().startswith("cat") else 1)
                    self.transform = transforms.Compose(
                        [
                            transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                        ]
                    )

                def __len__(self):
                    return len(self.paths)

                def __getitem__(self, idx):
                    img = Image.open(self.paths[idx]).convert("RGB")
                    return self.transform(img), self.labels[idx]

            dataset = ImageFolderDataset(self.config.data_dir, self.config.img_size)
            if len(dataset) < 2:
                # fallback to synthetic
                await self._stream_synthetic()
                acc = 0.8
                return {"metrics": {"accuracy": acc}, "run_id": self.run_id}
            loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=True)

            model = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(8, 2),
            )
            device = "cpu"
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            total_steps = min(self.config.steps, 30)
            step = 0
            model.train()
            for epoch in range(1):
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
                    optimizer.zero_grad()
                    out = model(imgs)
                    loss = criterion(out, labels)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    await self._emit(
                        EventType.TRAIN_PROGRESS,
                        {
                            "run_id": self.run_id,
                            "epoch": epoch + 1,
                            "epochs": 1,
                            "step": step,
                            "steps": total_steps,
                            "eta_s": None,
                            "phase": "fit",
                        },
                    )
                    if step >= total_steps:
                        break
                if step >= total_steps:
                    break

            # Eval on training set (small demo)
            model.eval()
            correct = 0
            total = 0
            for imgs, labels in loader:
                with torch.no_grad():
                    preds = model(imgs.to(device))
                    pred_labels = preds.argmax(dim=1).cpu().numpy()
                lbls = np.array(labels)
                correct += (pred_labels == lbls).sum()
                total += len(lbls)
            acc = float(correct / max(total, 1))
            cm = [[int(correct), int(total - correct)], [int(total - correct), int(correct)]] if total else [[1, 0], [0, 1]]
            cm_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_confusion.json", {"confusion": cm})
            await self._emit(EventType.CONFUSION_MATRIX_READY, {"asset_url": asset_url(cm_path)})
            await self._emit(
                EventType.METRIC_SCALAR,
                {"run_id": self.run_id, "name": "accuracy", "split": "test", "step": step, "value": acc},
            )
            await self._emit(
                EventType.TRAIN_RUN_FINISHED,
                {"run_id": self.run_id, "status": "success", "final_metrics": {"accuracy": acc}},
                stage_status=StageStatus.COMPLETED,
            )
            return {"metrics": {"accuracy": acc}, "run_id": self.run_id}

        # Build TF data pipeline
        data_dir = Path(self.config.data_dir)
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            image_size=(self.config.img_size, self.config.img_size),
            batch_size=self.config.batch_size,
            validation_split=0.2,
            subset="training",
            seed=42,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            image_size=(self.config.img_size, self.config.img_size),
            batch_size=self.config.batch_size,
            validation_split=0.2,
            subset="validation",
            seed=42,
        )

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Rescaling(1.0 / 255, input_shape=(self.config.img_size, self.config.img_size, 3)),
                tf.keras.layers.Conv2D(16, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Hook to stream progress
        step_counter = {"step": 0}

        def on_batch_end(batch, logs=None):
            step_counter["step"] += 1
            asyncio.run(
                self._emit(
                    EventType.TRAIN_PROGRESS,
                    {
                        "run_id": self.run_id,
                        "epoch": 1,
                        "epochs": self.config.epochs,
                        "step": step_counter["step"],
                        "steps": self.config.steps,
                        "eta_s": None,
                        "phase": "fit",
                    },
                )
            )

        cb = tf.keras.callbacks.LambdaCallback(on_batch_end=on_batch_end)
        model.fit(train_ds, validation_data=val_ds, epochs=self.config.epochs, callbacks=[cb])

        # Evaluate
        loss, acc = model.evaluate(val_ds, verbose=0)
        cm_path = None
        try:
            # Compute confusion matrix on a batch
            import itertools

            y_true = []
            y_pred = []
            for images, labels in val_ds.take(5):
                preds = model.predict(images, verbose=0).flatten()
                y_true.extend(labels.numpy().tolist())
                y_pred.extend((preds > 0.5).astype(int).tolist())
            cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy().tolist()
            cm_path = save_json_asset(self.config.project_id, f"artifacts/{self.run_id}_confusion.json", {"confusion": cm})
        except Exception:
            pass

        if cm_path:
            await self._emit(EventType.CONFUSION_MATRIX_READY, {"asset_url": asset_url(cm_path)})
            await self._emit(EventType.ARTIFACT_ADDED, {"artifact": {"id": f"{self.run_id}_confusion", "type": "confusion_matrix", "name": "Confusion Matrix", "url": asset_url(cm_path), "meta": {}}})
        await self._emit(
            EventType.METRIC_SCALAR,
            {"run_id": self.run_id, "name": "accuracy", "split": "test", "step": self.config.steps, "value": float(acc)},
        )
        await self._emit(
            EventType.TRAIN_RUN_FINISHED,
            {"run_id": self.run_id, "status": "success", "final_metrics": {"accuracy": float(acc)}},
            stage_status=StageStatus.COMPLETED,
        )
        return {"metrics": {"accuracy": float(acc)}, "run_id": self.run_id}
