import type { StepPhase } from '../types';
import { clamp01, seeded, formatClock } from '../types';

export interface LossPoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
}

export interface MetricPoint {
  epoch: number;
  value: number;
}

/**
 * Builds a synthetic loss series with exponential decay
 */
export function buildLossSeries(total: number): LossPoint[] {
  const pts: LossPoint[] = [];
  for (let i = 0; i < total; i += 1) {
    const base = 1.35 * Math.exp(-i / 8);
    const noise = (seeded(i + 1) - 0.5) * 0.06;
    const train_loss = Math.max(0.05, base + noise);
    const val_loss = Math.max(0.06, base * 1.05 + noise * 0.8);
    pts.push({ epoch: i + 1, train_loss, val_loss });
  }
  return pts;
}

/**
 * Builds a synthetic accuracy series with logarithmic growth
 */
export function buildAccuracySeries(total: number): MetricPoint[] {
  const pts: MetricPoint[] = [];
  for (let i = 0; i < total; i += 1) {
    const base = 0.45 + 0.5 * (1 - Math.exp(-i / 10));
    const noise = (seeded(i + 100) - 0.5) * 0.03;
    pts.push({ epoch: i + 1, value: clamp01(base + noise) });
  }
  return pts;
}

/**
 * Maps metric series to train/val split with wobble
 */
export function mapMetricSeries(
  series: MetricPoint[],
  metricKind: 'accuracy' | 'f1' | 'rmse'
) {
  return series.map((point, idx) => {
    const wobble = Math.sin(point.epoch / 4 + idx * 0.35);
    if (metricKind === 'rmse') {
      const trainMetric = Math.max(0, point.value * (0.92 + 0.02 * wobble));
      const valMetric = Math.max(0, point.value * (1.06 + 0.03 * wobble));
      return { epoch: point.epoch, trainMetric, valMetric };
    }
    const trainMetric = clamp01(point.value + 0.03 + 0.01 * wobble);
    const valMetric = clamp01(point.value - 0.03 + 0.008 * wobble);
    return { epoch: point.epoch, trainMetric, valMetric };
  });
}

/**
 * Computes the current phase based on elapsed time and phase weights
 */
export function computeWeightedPhase(
  phases: StepPhase[],
  elapsedMs: number,
  durationMs: number
) {
  const total = Math.max(1, durationMs);
  const weights = phases.map((p) => (typeof p.weight === 'number' ? p.weight : 1));
  const sum = weights.reduce((a, b) => a + b, 0) || 1;
  const normalized = weights.map((w) => w / sum);

  const durs = normalized.map((w) => Math.max(1, Math.round(total * w)));
  // Fix rounding drift by snapping last duration.
  const drift = durs.reduce((a, b) => a + b, 0) - total;
  durs[durs.length - 1] = Math.max(1, durs[durs.length - 1] - drift);

  let remaining = Math.max(0, elapsedMs);
  let idx = 0;
  while (idx < durs.length - 1 && remaining >= durs[idx]) {
    remaining -= durs[idx];
    idx += 1;
  }

  const phase = phases[idx];
  const phaseDuration = Math.max(1, durs[idx]);
  const phaseElapsed = Math.min(phaseDuration, remaining);
  const phaseProgress = clamp01(phaseElapsed / phaseDuration);

  return { phaseIndex: idx, phase, phaseProgress, phases, durs };
}

/**
 * Write JSON to a file via updateFileContent callback
 */
export function writeJson(
  updateFileContent: (path: string, content: string | ((prev: string) => string)) => void,
  path: string,
  value: unknown
) {
  updateFileContent(path, JSON.stringify(value, null, 2));
}

/**
 * Append a log line to the training log file
 */
export function appendLog(
  updateFileContent: (path: string, content: string | ((prev: string) => string)) => void,
  line: string,
  level: 'INFO' | 'WARN' | 'ERROR' = 'INFO'
) {
  updateFileContent('/logs/training.log', (prev) => `${prev || ''}[${formatClock()}] [${level}] ${line}\n`);
}
