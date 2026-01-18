import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  ArtifactAddedPayload,
  DatasetSampleReadyPayload,
  LogLinePayload,
  MetricScalarPayload,
  EventType,
  StageID,
} from '../lib/contract';
import type { EventEnvelope } from '../lib/ws';
import { useProjectStore } from '../store/projectStore';
import type { MetricsState } from '../mock/useMockAutoMLStream';

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
  thinkingByStage: {},
  datasetPreview: null,
};

export function useLiveMetrics() {
  const events = useProjectStore((s) => s.events);
  const [metricsState, setMetricsState] = useState<MetricsState>(emptyMetrics);
  const assetMetaRef = useRef(new Map<string, Record<string, unknown>>());

  const latestByType = useMemo(() => {
    const map = new Map<string, EventEnvelope>();
    for (const evt of events) {
      const key = evt.event?.name ?? evt.type ?? 'EVENT';
      map.set(key, evt);
    }
    return map;
  }, [events]);

  useEffect(() => {
    // Reduce all events into metrics state
    let next = metricsState;
    for (const evt of events.slice(metricsState.lossSeries.length ? 0 : 0)) {
      next = reduceMetrics(next, evt, assetMetaRef.current);
    }
    setMetricsState(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  return { metricsState, latestByType };
}

function reduceMetrics(prev: MetricsState, evt: EventEnvelope, assetMeta: Map<string, Record<string, unknown>>): MetricsState {
  const name = evt.event?.name as EventType | undefined;
  const payload = evt.event?.payload as Record<string, unknown> | undefined;

  if (name === 'ARTIFACT_ADDED') {
    const artifactPayload = payload as ArtifactAddedPayload | undefined;
    const artifact = artifactPayload?.artifact;
    if (artifact?.meta) {
      assetMeta.set(artifact.url || artifact.id, artifact.meta as Record<string, unknown>);
      const meta = artifact.meta as Record<string, unknown>;
      if (meta.kind === 'loss_surface' && meta.spec) return { ...prev, surfaceSpec: meta.spec as any };
      if (meta.kind === 'gradient_path' && meta.points) return { ...prev, gradientPath: meta.points as any };
      if (meta.kind === 'embedding_points' && meta.points) return { ...prev, embeddingPoints: meta.points as any };
      if (meta.kind === 'residuals' && meta.points) return { ...prev, residuals: meta.points as any };
      if (meta.kind === 'pipeline_graph' && meta.graph) return { ...prev, pipelineGraph: meta.graph as any };
      if (meta.kind === 'confusion_matrix' && meta.matrix) return { ...prev, confusionTable: meta.matrix as number[][] };
      if (meta.kind === 'feature_importance' && meta.ranking) {
        return { ...prev, leaderboard: [{ rank: 1, model: 'Top Features', metricName: 'importance', metricValue: 1, params: { ranking: meta.ranking } }] };
      }
    }
  }

  switch (name) {
    case 'LOG_LINE': {
      const text = (payload as LogLinePayload | undefined)?.text ?? '';
      const prefix = 'THINKING:';
      if (!text.startsWith(prefix)) return prev;
      const stage = evt.stage?.id as StageID | undefined;
      if (!stage) return prev;
      const msg = text.slice(prefix.length).trim();
      const existing = prev.thinkingByStage[stage] ?? [];
      return { ...prev, thinkingByStage: { ...prev.thinkingByStage, [stage]: [...existing, msg].slice(-200) } };
    }
    case 'METRIC_SCALAR': {
      const m = payload as MetricScalarPayload | undefined;
      if (!m) return prev;
      const epoch = m.step;
      if (m.name === 'loss' || m.name === 'train_loss' || m.name === 'val_loss') {
        const existing = prev.lossSeries.find((p) => p.epoch === epoch);
        const nextPoint = existing
          ? {
              ...existing,
              train_loss: m.split === 'train' ? m.value : existing.train_loss,
              val_loss: m.split === 'val' ? m.value : existing.val_loss,
            }
          : { epoch, train_loss: m.split === 'train' ? m.value : m.value, val_loss: m.split === 'val' ? m.value : m.value };
        const lossSeries = [...prev.lossSeries.filter((p) => p.epoch !== epoch), nextPoint].sort((a, b) => a.epoch - b.epoch);
        return { ...prev, lossSeries };
      }
      if (m.name === 'accuracy') {
        const accSeries = [...prev.accSeries.filter((p) => p.epoch !== epoch), { epoch, value: m.value }].sort((a, b) => a.epoch - b.epoch);
        return { ...prev, accSeries, metricsSummary: { ...prev.metricsSummary, accuracy: m.value } };
      }
      if (m.name === 'f1') {
        const f1Series = [...prev.f1Series.filter((p) => p.epoch !== epoch), { epoch, value: m.value }].sort((a, b) => a.epoch - b.epoch);
        return { ...prev, f1Series, metricsSummary: { ...prev.metricsSummary, f1: m.value } };
      }
      if (m.name === 'rmse') {
        const rmseSeries = [...prev.rmseSeries.filter((p) => p.epoch !== epoch), { epoch, value: m.value }].sort((a, b) => a.epoch - b.epoch);
        return { ...prev, rmseSeries, metricsSummary: { ...prev.metricsSummary, rmse: m.value } };
      }
      return prev;
    }
    case 'DATASET_SAMPLE_READY': {
      const sample = payload as DatasetSampleReadyPayload | any;
      if (sample?.images) {
        return {
          ...prev,
          datasetPreview: { rows: [], columns: [], dataType: 'image', imageData: { width: 0, height: 0, pixels: [], needsClientLoad: true } },
        };
      }
      if (sample?.columns && Array.isArray(sample.columns)) {
        return {
          ...prev,
          datasetPreview: {
            rows: sample.rows ?? [],
            columns: sample.columns as string[],
            dataType: 'tabular',
          },
        };
      }
      return prev;
    }
    case 'CONFUSION_MATRIX_READY': {
      const url = (payload as any)?.asset_url;
      const meta = url ? assetMeta.get(url) : undefined;
      if (meta?.matrix) {
        return { ...prev, confusionTable: meta.matrix as number[][] };
      }
      return prev;
    }
    case 'RESIDUALS_PLOT_READY': {
      const url = (payload as any)?.asset_url;
      const meta = url ? assetMeta.get(url) : undefined;
      if (meta?.points) {
        return { ...prev, residuals: meta.points as any };
      }
      return prev;
    }
    case 'LEADERBOARD_UPDATED': {
      const rows = (payload?.rows as Array<{ model: string; params: Record<string, unknown>; metric: number }>) ?? [];
      return {
        ...prev,
        leaderboard: rows.map((row, idx) => ({
          rank: idx + 1,
          model: row.model,
          metricName: 'metric',
          metricValue: row.metric,
          params: row.params ?? {},
        })),
      };
    }
    default:
      return prev;
  }
}
