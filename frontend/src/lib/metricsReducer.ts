import type { 
  ArtifactAddedPayload, 
  DatasetSampleReadyPayload, 
  LogLinePayload, 
  MetricScalarPayload, 
  StageID 
} from './contract';
import type { 
  EmbeddingPoint, 
  GradientPoint, 
  LossPoint, 
  MetricPoint, 
  ResidualPoint 
} from '../mock/scenarios/scenarioTypes';
import type { LossSurfaceSpec } from '../mock/backendEventTypes';
import type { EventEnvelope } from './ws';

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
  thinkingByStage: Partial<Record<StageID, string[]>>;
  datasetPreview: { 
    rows: number[][]; 
    columns: string[]; 
    dataType: 'tabular' | 'image'; 
    imageData?: { 
      width: number; 
      height: number; 
      pixels: Array<{ r: number; g: number; b: number }>; 
      needsClientLoad?: boolean 
    } 
  } | null;
}

export const emptyMetrics: MetricsState = {
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

export function reduceMetricsState(
  prev: MetricsState, 
  event: EventEnvelope, 
  assetMap: Map<string, Record<string, unknown>>
): MetricsState {
  const eventName = event.event?.name;
  const payload = event.event?.payload as Record<string, unknown> | undefined;

  if (eventName === 'ARTIFACT_ADDED') {
    const artifactPayload = payload as ArtifactAddedPayload | undefined;
    const artifact = artifactPayload?.artifact;
    if (artifact?.url && artifact.meta) {
      assetMap.set(artifact.url, artifact.meta as Record<string, unknown>);
      const meta = artifact.meta as Record<string, unknown>;
      if (meta.kind === 'loss_surface' && meta.spec) {
        return { ...prev, surfaceSpec: meta.spec as LossSurfaceSpec };
      }
      if (meta.kind === 'gradient_path' && meta.points) {
        return { ...prev, gradientPath: meta.points as GradientPoint[] };
      }
      if (meta.kind === 'embedding_points' && meta.points) {
        return { ...prev, embeddingPoints: meta.points as EmbeddingPoint[] };
      }
      if (meta.kind === 'residuals' && meta.points) {
        return { ...prev, residuals: meta.points as ResidualPoint[] };
      }
      if (meta.kind === 'pipeline_graph' && meta.nodes && meta.edges) {
        return { 
          ...prev, 
          pipelineGraph: { 
            nodes: meta.nodes as Array<{ id: string; label: string }>, 
            edges: meta.edges as Array<{ from: string; to: string }> 
          } 
        };
      }
    }
  }

  switch (eventName) {
    case 'LOG_LINE': {
      const logPayload = payload as LogLinePayload | undefined;
      const text = logPayload?.text ?? '';
      const prefix = 'THINKING:';
      if (!text.startsWith(prefix)) return prev;
      const stage = event.stage?.id as StageID | undefined;
      if (!stage) return prev;
      const message = text.slice(prefix.length).trim();
      const existing = prev.thinkingByStage[stage] ?? [];
      const next = [...existing, message].slice(-200);
      return { ...prev, thinkingByStage: { ...prev.thinkingByStage, [stage]: next } };
    }
    case 'METRIC_SCALAR': {
      const metricPayload = payload as MetricScalarPayload | undefined;
      if (!metricPayload) return prev;
      const epoch = metricPayload.step;
      if (metricPayload.name === 'loss') {
        const existing = prev.lossSeries.find((p) => p.epoch === epoch);
        const nextPoint: LossPoint = existing
          ? {
              ...existing,
              train_loss: metricPayload.split === 'train' ? metricPayload.value : existing.train_loss,
              val_loss: metricPayload.split === 'val' ? metricPayload.value : existing.val_loss,
            }
          : {
              epoch,
              train_loss: metricPayload.split === 'train' ? metricPayload.value : metricPayload.value,
              val_loss: metricPayload.split === 'val' ? metricPayload.value : metricPayload.value,
            };

        const nextLoss = [...prev.lossSeries.filter((p) => p.epoch !== epoch), nextPoint].sort((a, b) => a.epoch - b.epoch);
        return { ...prev, lossSeries: nextLoss };
      }

      if (metricPayload.name === 'accuracy') {
        const nextAcc = [...prev.accSeries.filter((p) => p.epoch !== epoch), { epoch, value: metricPayload.value }].sort(
          (a, b) => a.epoch - b.epoch
        );
        return { ...prev, accSeries: nextAcc, metricsSummary: { ...prev.metricsSummary, accuracy: metricPayload.value } };
      }

      if (metricPayload.name === 'f1') {
        const nextF1 = [...prev.f1Series.filter((p) => p.epoch !== epoch), { epoch, value: metricPayload.value }].sort(
          (a, b) => a.epoch - b.epoch
        );
        return { ...prev, f1Series: nextF1, metricsSummary: { ...prev.metricsSummary, f1: metricPayload.value } };
      }

      if (metricPayload.name === 'rmse') {
        const nextRmse = [...prev.rmseSeries.filter((p) => p.epoch !== epoch), { epoch, value: metricPayload.value }].sort(
          (a, b) => a.epoch - b.epoch
        );
        return { ...prev, rmseSeries: nextRmse, metricsSummary: { ...prev.metricsSummary, rmse: metricPayload.value } };
      }

      return prev;
    }
    case 'DATASET_SAMPLE_READY': {
      const samplePayload = payload as DatasetSampleReadyPayload | undefined;
      if (!samplePayload?.asset_url) return prev;
      const meta = assetMap.get(samplePayload.asset_url);
      if (!meta || meta.kind !== 'dataset_sample') return prev;
      return {
        ...prev,
        datasetPreview: {
          rows: (meta.rows as number[][]) ?? [],
          columns: (meta.columns as string[]) ?? [],
          dataType: (meta.data_type as 'tabular' | 'image') ?? 'tabular',
          imageData: meta.image_data as any,
        },
      };
    }
    case 'CONFUSION_MATRIX_READY': {
      const assetUrl = payload?.asset_url as string | undefined;
      const meta = assetUrl ? assetMap.get(assetUrl) : undefined;
      if (!meta || meta.kind !== 'confusion_matrix') return prev;
      return { ...prev, confusionTable: meta.matrix as number[][] };
    }
    case 'RESIDUALS_PLOT_READY': {
      const assetUrl = payload?.asset_url as string | undefined;
      const meta = assetUrl ? assetMap.get(assetUrl) : undefined;
      if (!meta || meta.kind !== 'residuals') return prev;
      return { ...prev, residuals: meta.points as ResidualPoint[] };
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
