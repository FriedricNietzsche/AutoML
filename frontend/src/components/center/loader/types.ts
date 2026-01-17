import type { LossSurfaceSpec } from '../../../mock/backendEventTypes';

export type VisualId =
  | 'neuralNetForward'
  | 'neuralNetBackprop'
  | 'gradDescent'
  | 'evaluation'
  | 'confusionMatrix'
  | 'embeddingScatter'
  | 'residuals';

export type StepPhase =
  | { kind: 'operation'; weight?: number }
  | { kind: 'graph'; graphType: 'loss' | 'accuracy'; weight?: number }
  | { kind: 'visual'; visualId: VisualId; weight?: number };

export type StepDef = {
  id: string;
  title: string;
  subtitle: string;
  equations: string[];
  durationMs: number;
  matrixLabel?: string;
  matrixRows?: number;
  matrixCols?: number;
  phases: StepPhase[];
};

export type VisualBaseProps = {
  timeMs: number;
  phaseProgress: number;
  seed: number;
  reducedMotion: boolean;
};

export type VisualProps = VisualBaseProps & {
  writeArtifact?: (path: string, value: unknown) => void;
  confusion?: number[][];
  metrics?: { accuracy: number; precision: number; recall: number; f1: number };
  curve?: { kind: 'pr' | 'roc'; points: Array<{ x: number; y: number }> };
  points?: Array<{ id: number; x: number; y: number; label: number; weight: number }>;
  path?: Array<{ x: number; y: number }>;
  residuals?: Array<{ pred: number; true: number; residual: number }>;
  showConfusion?: boolean;
  surfaceSpec?: LossSurfaceSpec | null;
};

export const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export const seeded = (n: number) => {
  const x = Math.sin(n * 9999) * 10000;
  return x - Math.floor(x);
};

export const formatClock = () => {
  const t = new Date();
  return t.toISOString().split('T')[1].slice(0, 8);
};
