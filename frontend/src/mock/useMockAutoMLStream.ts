import { useEffect, useMemo, useRef, useState } from 'react';
import type { BackendEvent, LossSurfaceSpec } from './backendEventTypes';
import { createMockAutoMLStream } from './mockBackendStream';
import type { ScenarioId } from './scenarios';
import type { EmbeddingPoint, GradientPoint, LossPoint, MetricPoint, ResidualPoint } from './scenarios/scenarioTypes';

export interface MetricsState {
  lossSeries: LossPoint[];
  accSeries: MetricPoint[];
  f1Series: MetricPoint[];
  rmseSeries: MetricPoint[];
  confusionTable: number[][] | null;
  embeddingPoints: EmbeddingPoint[];
  gradientPath: GradientPoint[];
  residuals: ResidualPoint[];
  surfaceSpec: LossSurfaceSpec | null;
  pipelineGraph: { nodes: Array<{ id: string; label: string }>; edges: Array<{ from: string; to: string }> } | null;
  leaderboard: Array<{ rank: number; model: string; metricName: string; metricValue: number; params: Record<string, unknown> }>;
  metricsSummary: { accuracy?: number; f1?: number; rmse?: number };
}

const emptyMetrics: MetricsState = {
  lossSeries: [],
  accSeries: [],
  f1Series: [],
  rmseSeries: [],
  confusionTable: null,
  embeddingPoints: [],
  gradientPath: [],
  residuals: [],
  surfaceSpec: null,
  pipelineGraph: null,
  leaderboard: [],
  metricsSummary: {},
};

export function useMockAutoMLStream({ scenarioId, seed, enabled }: { scenarioId: ScenarioId; seed?: number; enabled?: boolean }) {
  const [events, setEvents] = useState<BackendEvent[]>([]);
  const [metricsState, setMetricsState] = useState<MetricsState>(emptyMetrics);
  const latestByType = useMemo(() => {
    const map = new Map<string, BackendEvent>();
    for (const event of events) map.set(event.type, event);
    return map;
  }, [events]);

  const runIdRef = useRef(0);
  const pendingEventsRef = useRef<BackendEvent[]>([]);
  const pendingMetricsRef = useRef<MetricsState>(emptyMetrics);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled) return;
    runIdRef.current += 1;
    const runId = runIdRef.current;
    setEvents([]);
    setMetricsState(emptyMetrics);
    pendingEventsRef.current = [];
    pendingMetricsRef.current = emptyMetrics;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const stream = createMockAutoMLStream({ scenarioId, seed });

    const flush = () => {
      rafRef.current = null;
      const batch = pendingEventsRef.current;
      if (batch.length === 0) return;
      pendingEventsRef.current = [];
      const nextMetrics = pendingMetricsRef.current;
      setEvents((prev) => [...prev, ...batch]);
      setMetricsState(nextMetrics);
    };

    (async () => {
      for await (const event of stream) {
        if (runIdRef.current !== runId) return;
        pendingEventsRef.current.push(event);
        pendingMetricsRef.current = reduceMetricsState(pendingMetricsRef.current, event);
        if (rafRef.current === null) {
          rafRef.current = requestAnimationFrame(flush);
        }
      }
    })();

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [enabled, scenarioId, seed]);

  return { events, latestByType, metricsState };
}

function reduceMetricsState(prev: MetricsState, event: BackendEvent): MetricsState {
  switch (event.type) {
    case 'METRIC_SCALAR': {
      if (event.metric === 'train_loss' || event.metric === 'val_loss') {
        const existing = prev.lossSeries.find((p) => p.epoch === event.epoch);
        const nextPoint: LossPoint = existing
          ? {
              ...existing,
              train_loss: event.metric === 'train_loss' ? event.value : existing.train_loss,
              val_loss: event.metric === 'val_loss' ? event.value : existing.val_loss,
            }
          : {
              epoch: event.epoch,
              train_loss: event.metric === 'train_loss' ? event.value : event.value,
              val_loss: event.metric === 'val_loss' ? event.value : event.value,
            };

        const nextLoss = [...prev.lossSeries.filter((p) => p.epoch !== event.epoch), nextPoint].sort(
          (a, b) => a.epoch - b.epoch
        );
        return { ...prev, lossSeries: nextLoss };
      }

      if (event.metric === 'accuracy') {
        const nextAcc = [...prev.accSeries.filter((p) => p.epoch !== event.epoch), { epoch: event.epoch, value: event.value }].sort(
          (a, b) => a.epoch - b.epoch
        );
        return { ...prev, accSeries: nextAcc, metricsSummary: { ...prev.metricsSummary, accuracy: event.value } };
      }

      if (event.metric === 'f1') {
        const nextF1 = [...prev.f1Series.filter((p) => p.epoch !== event.epoch), { epoch: event.epoch, value: event.value }].sort(
          (a, b) => a.epoch - b.epoch
        );
        return { ...prev, f1Series: nextF1, metricsSummary: { ...prev.metricsSummary, f1: event.value } };
      }

      if (event.metric === 'rmse') {
        const nextRmse = [...prev.rmseSeries.filter((p) => p.epoch !== event.epoch), { epoch: event.epoch, value: event.value }].sort(
          (a, b) => a.epoch - b.epoch
        );
        return { ...prev, rmseSeries: nextRmse, metricsSummary: { ...prev.metricsSummary, rmse: event.value } };
      }

      return prev;
    }
    case 'METRIC_TABLE':
      if (event.table === 'confusion') return { ...prev, confusionTable: event.data };
      return prev;
    case 'PIPELINE_GRAPH':
      return { ...prev, pipelineGraph: { nodes: event.nodes, edges: event.edges } };
    case 'LEADERBOARD_UPDATED':
      return { ...prev, leaderboard: event.entries };
    case 'LOSS_SURFACE_SPEC':
      return { ...prev, surfaceSpec: event.spec };
    case 'GD_PATH':
      return { ...prev, gradientPath: event.points };
    case 'ARTIFACT_WRITTEN': {
      if (event.path === 'artifacts/embedding_points.json') {
        try {
          const points = JSON.parse(event.content) as EmbeddingPoint[];
          return { ...prev, embeddingPoints: points };
        } catch {
          return prev;
        }
      }
      if (event.path === 'artifacts/gradient_path.json') {
        try {
          const parsed = JSON.parse(event.content) as { path: GradientPoint[] };
          const path = parsed?.path ?? [];
          return { ...prev, gradientPath: path };
        } catch {
          return prev;
        }
      }
      if (event.path === 'artifacts/residuals.json') {
        try {
          const parsed = JSON.parse(event.content) as { points: ResidualPoint[] };
          return { ...prev, residuals: parsed.points ?? [] };
        } catch {
          return prev;
        }
      }
      return prev;
    }
    default:
      return prev;
  }
}
