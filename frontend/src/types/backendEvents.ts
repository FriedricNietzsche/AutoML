export type StepId = 'S0' | 'S1' | 'S2' | 'S3' | 'S4' | 'S5' | 'S6' | 'S7';

export type StepStatus = 'pending' | 'running' | 'waiting_confirmation' | 'completed' | 'failed';

export type LogLevel = 'info' | 'warn' | 'error';

export interface BaseEvent<T extends string, P> {
  type: T;
  ts: number;
  step: StepId;
  payload: P;
}

export interface PlanVariant {
  id: string;
  title: string;
  summary: string;
  expectedRuntimeMin?: number;
  interpretability?: string;
  details?: Record<string, unknown>;
}

export interface DatasetColumn {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'category' | 'datetime';
  missingPct: number; // 0-1
}

export interface DatasetResult {
  id: string;
  name: string;
  license: string;
  sizeMB: number;
  rows: number;
  columns: DatasetColumn[];
  description: string;
}

export interface PipelineGraph {
  nodes: Array<{ id: string; label: string; kind: string }>;
  edges: Array<{ from: string; to: string }>;
}

export interface FeatureSummary {
  totalFeatures: number;
  topFeatures: Array<{ name: string; importance: number }>;
  encoding: string[];
}

export interface LeaderboardEntry {
  modelId: string;
  modelName: string;
  metric: number;
  params?: Record<string, unknown>;
  rank: number;
}

export type BackendEvent =
  | BaseEvent<'STEP_STATUS', { status: StepStatus; progress?: number; message?: string }>
  | BaseEvent<'PLAN_PROPOSED', { plans: PlanVariant[]; recommendedPlanId?: string }>
  | BaseEvent<'PLAN_SELECTED', { planId: string; reason?: string }>
  | BaseEvent<'DATASET_SEARCH_RESULTS', { query: string; results: DatasetResult[] }>
  | BaseEvent<'DATASET_INGESTED', { datasetId: string; rows: number; columns: number; samplePath?: string }>
  | BaseEvent<'PROFILE_PROGRESS', { stage: string; progress: number }>
  | BaseEvent<'PROFILE_SUMMARY', { rows: number; columns: number; missingness: Array<{ column: string; missingPct: number }>; target?: string }>
  | BaseEvent<'PIPELINE_GRAPH', { graph: PipelineGraph }>
  | BaseEvent<'FEATURE_SUMMARY', { summary: FeatureSummary }>
  | BaseEvent<'FILE_UPDATED', { path: string; message: string }>
  | BaseEvent<'LEADERBOARD_UPDATED', { metric: string; entries: LeaderboardEntry[] }>
  | BaseEvent<'TRAIN_PROGRESS', { runId: string; modelName: string; progress: number; etaSec?: number }>
  | BaseEvent<'METRIC_SCALAR', { name: string; value: number; stepIndex?: number; runId?: string }>
  | BaseEvent<'METRIC_TABLE', { name: string; columns: string[]; rows: Array<Record<string, string | number>> }>
  | BaseEvent<'RESOURCE_STATS', { cpuPct: number; ramMB: number; gpuPct?: number }>
  | BaseEvent<'LOG_LINE', { level: LogLevel; message: string }>
  | BaseEvent<'ARTIFACT_WRITTEN', { path: string; content: string; contentType?: 'json' | 'text' | 'binary' }>
  | BaseEvent<'REPORT_READY', { path: string; highlights: string[]; verificationNotes: string[] }>
  | BaseEvent<'EXPORT_READY', { targets: string[]; files: string[]; notes?: string }>;
