# AutoML Frontend Visualization Fix

## Problem

The visualizations show all zeros because:
1. `TrainingLoader` expects either mock data or live metrics
2. When `useMockStream = false` (default), it uses `emptyMetricsState` with all empty arrays
3. The projectStore receives events but doesn't compute metrics from them

## Solution Options

### Option 1: Use Dashboard Tab (Recommended)
The `DashboardPane` is designed for live data visualization. It's already wired to projectStore.

**User Action**: Click the "Dashboard" tab to see live data instead of staying on "Preview"

### Option 2: Build Live Metrics from projectStore Events
Create a hook that processes projectStore events into metrics format.

### Option 3: Emit Metrics During Training
Backend needs to emit `METRIC_SCALAR` events during training so frontend can plot them.

## Current Status

- ✅ Backend emits events: `STAGE_STATUS`, `DATASET_SELECTED`, etc.
- ✅ Frontend receives events in projectStore
- ❌ TrainingLoader expects formatted metrics (lossSeries, accSeries)
- ❌ Backend doesn't emit `METRIC_SCALAR` during image training

## What Needs to Happen

For real visualizations in TrainingLoader:

1. **Backend**: Emit `METRIC_SCALAR` events during training
   ```python
   await event_bus.publish_event(
       project_id=project_id,
       event_name=EventType.METRIC_SCALAR,
       payload={
           "step": epoch,
           "name": "loss",
           "split": "train",
           "value": loss_value
       }
   )
   ```

2. **Frontend**: Convert METRIC_SCALAR events to series
   ```typescript
   const lossSeries = projectStore.trainingMetrics?.metricsHistory
     .filter(m => m.name === 'loss')
     .map(m => ({x: m.step, y: m.value}));
   ```

3. **Or**: Use DashboardPane which handles this automatically

## Recommendation

**Short term**: Use Dashboard tab for visualizations  
**Long term**: Implement real-time metrics emission from training loops
