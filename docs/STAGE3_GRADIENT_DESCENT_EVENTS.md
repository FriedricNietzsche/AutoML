# Stage 3 Gradient Descent Events (Backend → Frontend)

This document defines the additional **Stage 3 (TRAINING)** event payloads needed to drive the `GradientDescentViz` visualization.

The visualization needs **two inputs**:

1. A **loss surface definition** (parametric surface spec, or an asset URL to a sampled grid).
2. A **gradient descent path** (a list of 2D points streamed over time).

These events are designed to fit the existing envelope used by the backend event system:

```json
{
  "v": 1,
  "type": "EVENT",
  "project_id": "...",
  "seq": 123,
  "ts": 1730000000,
  "stage": { "id": "TRAIN", "index": 3, "status": "running" },
  "event": {
    "name": "EVENT_NAME",
    "payload": { }
  }
}
```

Only the `event.name` and `event.payload` additions are specified below.

---

## Coordinate Conventions (IMPORTANT)

The frontend currently maps incoming path points like this:

- Backend sends normalized points: `x_norm`, `y_norm` (recommended range `[-1, 1]`).
- Frontend converts to surface coordinates:
  - `x = x_norm * domainHalf`
  - `y = y_norm * domainHalf`

So the backend must send **`domainHalf`** (once) and then stream normalized `{x, y}` points.

---

## 1) Loss Surface Definition Events

### Event: `LOSS_SURFACE_SPEC_READY`
Send once per training run (typically alongside `TRAIN_RUN_STARTED`).

Payload:

```json
{
  "run_id": "string",
  "surface_spec": {
    "kind": "fixed_example | bowl | multi_hill | ripples",
    "params": { "any": "object depending on kind" },

    "domainHalf": 6.0,
    "zScale": 0.3
  }
}
```

Notes:

- `domainHalf` controls how path points are scaled into the surface domain.
- `zScale` controls vertical scaling in the 3D view.
- If omitted, the frontend may fall back to defaults, but backend-provided values keep everything consistent.

### (Optional alternative) Event: `LOSS_SURFACE_GRID_READY`
Use this if you want to render a real sampled loss surface (instead of a parametric function).

Payload:

```json
{
  "run_id": "string",
  "asset_url": "string",
  "grid_format": "float32",
  "resolution": 50,
  "domainHalf": 6.0,
  "zScale": 0.3
}
```

Notes:

- The asset should contain the grid values and the frontend should know how to decode it.
- This is more work than `LOSS_SURFACE_SPEC_READY` but can represent non-parametric surfaces.

---

## 2) Gradient Descent Path Streaming Events

### Event: `GD_PATH_STARTED`
Send once per training run, before `GD_PATH_UPDATE` messages.

Payload:

```json
{
  "run_id": "string",
  "space": "normalized",
  "domainHalf": 6.0,
  "point0": { "x": 0.15, "y": -0.42 }
}
```

Notes:

- `space` is explicitly declared to avoid ambiguity.
- `domainHalf` may be repeated here for convenience/robustness.

### Event: `GD_PATH_UPDATE` (preferred: batched)
Stream during training. Send batches every N points (e.g., 5–20) to avoid spamming.

Payload:

```json
{
  "run_id": "string",
  "space": "normalized",
  "step_start": 0,
  "points": [
    { "x": 0.15, "y": -0.42 },
    { "x": 0.14, "y": -0.40 },
    { "x": 0.13, "y": -0.37 }
  ]
}
```

Frontend behavior expectation:

- The UI appends these points to its `path` array.
- The visualization animates to the latest point as training progresses.

### Event: `GD_PATH_FINISHED`
Send once when the path is done.

Payload:

```json
{
  "run_id": "string",
  "final_step": 87,
  "reason": "converged | max_steps | cancelled | error"
}
```

---

## (Optional) Hyperparameters / Debug Info

### Event: `GD_HYPERPARAMS`
Useful if you want to show these values in the UI.

Payload:

```json
{
  "run_id": "string",
  "eta": 0.05,
  "steps": 100,
  "tolGrad": 0.001,
  "tolStep": 0.0005
}
```

---

## Backend Implementation Notes

### When to emit

- Emit `LOSS_SURFACE_SPEC_READY` at the start of Stage 3, ideally right after `TRAIN_RUN_STARTED`.
- Emit `GD_PATH_STARTED` right before the first `GD_PATH_UPDATE`.
- Emit `GD_PATH_UPDATE` periodically during training.
- Emit `GD_PATH_FINISHED` when the optimizer stops (or training finishes).

### Two practical approaches

1) **Illustrative (easy / recommended initially):**
   - Send a parametric `surface_spec` (e.g., `bowl`, `ripples`, `multi_hill`).
   - Generate a synthetic GD path that roughly matches training progress.

2) **Real-ish landscape (harder):**
   - If the true model is high-dimensional, pick a 2D projection (e.g., PCA directions in weight space).
   - Evaluate loss on a grid along those 2 directions and map optimizer steps into that plane.

---

## Minimal Checklist

To implement the visualization in a backend-driven way, the backend must provide:

- `LOSS_SURFACE_SPEC_READY` (or `LOSS_SURFACE_GRID_READY`)
- `GD_PATH_STARTED`
- `GD_PATH_UPDATE` (streamed)
- `GD_PATH_FINISHED`
