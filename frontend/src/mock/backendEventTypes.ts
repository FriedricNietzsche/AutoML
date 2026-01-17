export type StepId = 'S0' | 'S1' | 'S2' | 'S3' | 'S4' | 'S5' | 'S6' | 'S7';

export type LossSurfaceSpec =
  | { kind: 'fixed_example'; domainHalf: number; zScale: number }
  | {
      kind: 'bowl';
      domainHalf: number;
      zScale: number;
      params: { a: number; b: number; tiltX: number; tiltY: number; offset: number };
    }
  | {
      kind: 'ripples';
      domainHalf: number;
      zScale: number;
      params: { amp: number; freq: number; decay: number; bowlStrength: number; offset: number };
    }
  | {
      kind: 'multi_hill';
      domainHalf: number;
      zScale: number;
      params: { hills: Array<{ x: number; y: number; amp: number; sigma: number }>; bowlStrength: number; offset: number };
    };

export type BackendEvent =
  | { type: 'LOSS_SURFACE_SPEC'; spec: LossSurfaceSpec; ts: number }
  | { type: 'GD_PATH'; points: Array<{ x: number; y: number }>; ts: number }
  | {
      type: 'MODEL_THINKING';
      step: StepId;
      messages: string[];
      ts: number;
    }
  | {
      type: 'STEP_STATUS';
      step: StepId;
      status: 'waiting' | 'running' | 'complete';
      progress?: number;
      message?: string;
      ts: number;
    }
  | {
      type: 'PLAN_PROPOSED';
      step: 'S0' | 'S3' | 'S4';
      planId: string;
      variants: Array<{
        id: string;
        title: string;
        description: string;
        models: string[];
        expectedMinutes: number;
      }>;
      ts: number;
    }
  | {
      type: 'PLAN_SELECTED';
      step: 'S0' | 'S3' | 'S4';
      planId: string;
      variantId: string;
      ts: number;
    }
  | { type: 'TRAIN_PROGRESS'; epoch: number; totalEpochs: number; ts: number }
  | {
      type: 'METRIC_SCALAR';
      metric: 'train_loss' | 'val_loss' | 'accuracy' | 'f1' | 'rmse';
      epoch: number;
      value: number;
      split?: 'train' | 'val';
      ts: number;
    }
  | {
      type: 'METRIC_TABLE';
      table: 'confusion' | 'missingness';
      rows: string[];
      cols: string[];
      data: number[][];
      ts: number;
    }
  | {
      type: 'LEADERBOARD_UPDATED';
      entries: Array<{
        rank: number;
        model: string;
        metricName: string;
        metricValue: number;
        params: Record<string, unknown>;
      }>;
      ts: number;
    }
  | {
      type: 'PIPELINE_GRAPH';
      nodes: Array<{ id: string; label: string }>;
      edges: Array<{ from: string; to: string }>;
      ts: number;
    }
  | { type: 'LOG_LINE'; level: 'INFO' | 'WARN' | 'ERROR'; message: string; ts: number }
  | { type: 'ARTIFACT_WRITTEN'; path: string; content: string; ts: number }
  | { type: 'REPORT_READY'; summary: string; ts: number }
  | { type: 'EXPORT_READY'; files: string[]; ts: number }
  | {
      type: 'DATASET_SEARCH_RESULTS';
      query: string;
      results: Array<{ id: string; name: string; license: string; sizeMB: number; columns: number }>;
      ts: number;
    }
  | { type: 'DATASET_INGESTED'; datasetId: string; rows: number; columns: number; ts: number }
  | { type: 'DATASET_PREVIEW'; rows: number[][]; columns: string[]; dataType: 'tabular' | 'image'; imageData?: { width: number; height: number; pixels: Array<{ r: number; g: number; b: number }>; needsClientLoad?: boolean }; ts: number }
  | { type: 'PROFILE_PROGRESS'; stage: string; progress: number; ts: number }
  | {
      type: 'PROFILE_SUMMARY';
      rows: number;
      columns: number;
      missingness: Array<{ column: string; missingPct: number }>;
      ts: number;
    }
  | {
      type: 'FEATURE_SUMMARY';
      totalFeatures: number;
      topFeatures: Array<{ name: string; importance: number }>;
      ts: number;
    }
  | {
      type: 'RESOURCE_STATS';
      cpuPct: number;
      ramMB: number;
      gpuPct?: number;
      ts: number;
    };
