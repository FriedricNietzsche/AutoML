export type ScenarioId = 'A' | 'B' | 'C';

export type ScenarioTask = 'binary' | 'multiclass' | 'regression';

export type LossPoint = { epoch: number; train_loss: number; val_loss: number };
export type MetricPoint = { epoch: number; value: number };

export type EmbeddingPoint = { id: number; x: number; y: number; label: number; weight: number };
export type GradientPoint = { x: number; y: number };
export type ResidualPoint = { pred: number; true: number; residual: number };

export type PipelineGraph = {
  nodes: Array<{ id: string; label: string }>;
  edges: Array<{ from: string; to: string }>;
};

export type LeaderboardEntry = {
  rank: number;
  model: string;
  metricName: string;
  metricValue: number;
  params: Record<string, unknown>;
};

export type ScenarioData = {
  id: ScenarioId;
  name: string;
  model: string;
  task: ScenarioTask;
  totalEpochs: number;
  lossCurve: LossPoint[];
  accCurve?: MetricPoint[];
  rmseCurve?: MetricPoint[];
  confusion?: number[][];
  embeddingPoints: EmbeddingPoint[];
  gradientPath: GradientPoint[];
  residuals?: ResidualPoint[];
  leaderboard: LeaderboardEntry[];
  pipelineGraph: PipelineGraph;
};

export type ScenarioBuilder = (seed?: number) => ScenarioData;

export function createRng(seed = 1337) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function clamp(num: number, min: number, max: number) {
  return Math.max(min, Math.min(max, num));
}
