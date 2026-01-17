import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import clsx from 'clsx';

import type { StepDef, StepPhase, VisualId } from './types';
import { clamp01, formatClock, seeded } from './types';
import { VISUAL_LABEL, VISUALS } from './visuals/visualRegistry';
import TrainingLossVisualizer from './TrainingLossVisualizer';
import ModelMetricsVisualizer from './ModelMetricsVisualizer';
import { useMockAutoMLStream } from '../../../mock/useMockAutoMLStream';
import type { ScenarioId } from '../../../mock/scenarios';
import type { BackendEvent } from '../../../mock/backendEventTypes';
import { SCENARIO_VIZ, type LoaderStepId } from '../../../mock/scenarioVizConfig';

interface MetricPoint {
  epoch: number;
  value: number;
}

interface LossPoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
}

function mapMetricSeries(series: MetricPoint[], metricKind: 'accuracy' | 'f1' | 'rmse') {
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

export interface TrainingLoaderProps {
  onComplete: () => void;
  updateFileContent: (path: string, content: string | ((prev: string) => string)) => void;
  scenarioId?: ScenarioId;
  seed?: number;
  useMockStream?: boolean;
}

const DEFAULT_SEED = 1337;
const FRAME_DURATION_MS = 5000;
const EVAL_DURATION_MS = 9000;
const CONFUSION_DURATION_MS = 7000;

const STEP_TEMPLATES: Record<LoaderStepId, Omit<StepDef, 'id' | 'title' | 'subtitle' | 'durationMs'>> = {
  neuralNet: {
    equations: ['\\mathbf{h}=\\sigma(\\mathbf{W}\\mathbf{x}+\\mathbf{b})'],
    phases: [{ kind: 'visual', visualId: 'neuralNetForward' }],
  },
  matrixOps: {
    equations: ['\\mathbf{Y}=\\mathbf{X}\\mathbf{W}'],
    matrixLabel: 'X·W',
    matrixRows: 6,
    matrixCols: 8,
    phases: [{ kind: 'operation' }],
  },
  gradientDescent: {
    equations: ['\\theta_{t+1}=\\theta_t-\\eta\\nabla L(\\theta_t)'],
    phases: [{ kind: 'visual', visualId: 'gradDescent' }],
  },
  trainLoss: {
    equations: ['\\mathcal{L}(\\theta)'],
    phases: [{ kind: 'graph', graphType: 'loss' }],
  },
  modelMetric: {
    equations: ['\\mathrm{metric}(\\hat{y},y)'],
    phases: [{ kind: 'graph', graphType: 'accuracy' }],
  },
  embedding: {
    equations: ['\\mathbf{z}=f(\\mathbf{x})'],
    phases: [{ kind: 'visual', visualId: 'embeddingScatter' }],
  },
  evaluation: {
    equations: ['\\mathrm{F1}=2\\frac{PR}{P+R}'],
    phases: [{ kind: 'visual', visualId: 'evaluation' }],
  },
  residuals: {
    equations: ['r = y-\\hat{y}'],
    phases: [{ kind: 'visual', visualId: 'residuals' }],
  },
  confusionMatrix: {
    equations: ['\\mathbf{C}'],
    phases: [{ kind: 'visual', visualId: 'confusionMatrix' }],
  },
};

const phaseTitle = (phase: StepPhase) => {
  if (phase.kind === 'operation') return 'Operation';
  if (phase.kind === 'graph') return phase.graphType === 'loss' ? 'Graph (Loss)' : 'Graph (Accuracy)';
  return `Visual (${VISUAL_LABEL[phase.visualId]})`;
};

const getPhaseKindLabel = (phase: StepPhase) => {
  if (phase.kind === 'operation') return 'Operation';
  if (phase.kind === 'graph') return 'Graph';
  return 'Visual';
};

function buildLossSeries(total: number): LossPoint[] {
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

function buildAccuracySeries(total: number): MetricPoint[] {
  const pts: MetricPoint[] = [];
  for (let i = 0; i < total; i += 1) {
    const base = 0.45 + 0.5 * (1 - Math.exp(-i / 10));
    const noise = (seeded(i + 100) - 0.5) * 0.03;
    pts.push({ epoch: i + 1, value: clamp01(base + noise) });
  }
  return pts;
}

function writeJson(updateFileContent: TrainingLoaderProps['updateFileContent'], path: string, value: unknown) {
  updateFileContent(path, JSON.stringify(value, null, 2));
}

function appendLog(updateFileContent: TrainingLoaderProps['updateFileContent'], line: string, level: 'INFO' | 'WARN' | 'ERROR' = 'INFO') {
  updateFileContent('/logs/training.log', (prev) => `${prev || ''}[${formatClock()}] [${level}] ${line}\n`);
}

function MatrixGrid({
  label,
  rows,
  cols,
  timeMs,
  reducedMotion,
}: {
  label: string;
  rows: number;
  cols: number;
  timeMs: number;
  reducedMotion: boolean;
}) {
  const cells = useMemo(() => {
    const list: number[] = [];
    const total = rows * cols;
    for (let i = 0; i < total; i += 1) list.push(seeded(i + rows * 13 + cols * 7));
    return list;
  }, [rows, cols]);

  const opIndex = reducedMotion ? 0 : Math.floor(timeMs / 110) % Math.max(1, rows * cols);
  const opRow = Math.floor(opIndex / cols);
  const opCol = opIndex % cols;

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-mono text-replit-textMuted">
          {label}{' '}
          <span className="px-1.5 py-0.5 rounded border border-replit-border/60 bg-replit-surface/40">
            {rows}×{cols}
          </span>
        </div>
        <div className="text-[11px] text-replit-textMuted">Matrix op</div>
      </div>

      <div
        className="grid gap-px rounded-lg border border-replit-border/60 bg-replit-border/60 p-px overflow-hidden"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {cells.map((val, idx) => {
          const r = Math.floor(idx / cols);
          const c = idx % cols;

          const inOpRow = r === opRow;
          const inOpCol = c === opCol;
          const isOpCell = inOpRow && inOpCol;

          const wave = reducedMotion ? 0 : Math.sin(timeMs / 240 + idx * 0.35) * 0.08;
          const opBoost = reducedMotion ? 0 : inOpRow || inOpCol ? 0.12 : 0;
          const hotBoost = reducedMotion ? 0 : isOpCell ? 0.18 : 0;

          const dyn = clamp01(val + wave + opBoost + hotBoost);
          const display = (dyn * 2 - 1) * 1.15;

          return (
            <motion.div
              key={idx}
              initial={false}
              animate={
                reducedMotion
                  ? undefined
                  : {
                      opacity: 1,
                      scale: isOpCell ? 1.02 : 1,
                    }
              }
              transition={reducedMotion ? undefined : { type: 'spring', stiffness: 320, damping: 28 }}
              className={clsx(
                'flex items-center justify-center font-mono select-none',
                'h-10 md:h-12',
                'text-xs md:text-sm',
                'bg-replit-surface/40 text-replit-text',
                (inOpRow || inOpCol) && 'bg-replit-surfaceHover/60 ring-1 ring-replit-accent/20',
                isOpCell && 'ring-2 ring-replit-accent/70'
              )}
              style={{ opacity: 0.5 + dyn * 0.4 }}
            >
              {display.toFixed(2)}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

function computeWeightedPhase(phases: StepPhase[], elapsedMs: number, durationMs: number) {
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

export default function TrainingLoader({ onComplete, updateFileContent, scenarioId, seed, useMockStream = true }: TrainingLoaderProps) {
  const reducedMotionPref = useReducedMotion();
  const reducedMotion = !!reducedMotionPref;

  const [activeScenario, setActiveScenario] = useState<ScenarioId>(scenarioId ?? 'A');
  useEffect(() => {
    if (!scenarioId) return;
    setActiveScenario(scenarioId);
  }, [scenarioId]);

  const { events, metricsState } = useMockAutoMLStream({
    scenarioId: activeScenario,
    seed,
    enabled: useMockStream,
  });

  const scenarioConfig = useMemo(() => SCENARIO_VIZ[activeScenario], [activeScenario]);
  const steps = useMemo<StepDef[]>(() => {
    const base = scenarioConfig.steps.filter((s) => s.enabled);
    const confusionSize = metricsState.confusionTable?.length ?? 0;
    const shouldSplitConfusion = scenarioConfig.showConfusionMatrix && confusionSize > 2;

    const expanded: Array<{ id: LoaderStepId; title: string; subtitle: string; enabled: boolean }> = [];
    for (const stepDef of base) {
      expanded.push(stepDef);
      if (stepDef.id === 'evaluation' && shouldSplitConfusion) {
        expanded.push({
          id: 'confusionMatrix',
          title: 'Confusion Matrix',
          subtitle: 'Detailed class breakdown',
          enabled: true,
        });
      }
    }

    return expanded.map((s) => ({
      id: s.id,
      title: s.title,
      subtitle: s.subtitle,
      durationMs: s.id === 'evaluation' ? EVAL_DURATION_MS : s.id === 'confusionMatrix' ? CONFUSION_DURATION_MS : FRAME_DURATION_MS,
      ...STEP_TEMPLATES[s.id],
    }));
  }, [metricsState.confusionTable, scenarioConfig]);

  const metricKind = scenarioConfig.metricKind;

  const updateFileContentRef = useRef(updateFileContent);
  useEffect(() => {
    updateFileContentRef.current = updateFileContent;
  }, [updateFileContent]);

  const processedEventsRef = useRef(0);
  useEffect(() => {
    if (!useMockStream) return;
    if (events.length <= processedEventsRef.current) return;

    for (let i = processedEventsRef.current; i < events.length; i += 1) {
      const event = events[i] as BackendEvent;
      if (event.type === 'ARTIFACT_WRITTEN') {
        updateFileContentRef.current(event.path, event.content);
      }
      if (event.type === 'LOG_LINE') {
        appendLog(updateFileContentRef.current, event.message, event.level);
      }
    }

    processedEventsRef.current = events.length;
  }, [events, useMockStream]);

  const [now, setNow] = useState(0);
  const nowRef = useRef(0);

  const [stepIndex, setStepIndex] = useState(0);
  const [stepStartedAt, setStepStartedAt] = useState(0);
  const stepIndexRef = useRef(0);
  useEffect(() => {
    stepIndexRef.current = stepIndex;
  }, [stepIndex]);

  const completedRef = useRef(false);
  const advanceGuardRef = useRef(-1);
  const clockInitRef = useRef(false);

  const lossFull = useMemo(() => {
    if (useMockStream && metricsState.lossSeries.length > 0) return metricsState.lossSeries;
    return buildLossSeries(36);
  }, [metricsState.lossSeries, useMockStream]);

  const accFull = useMemo(() => {
    if (useMockStream && metricsState.accSeries.length > 0) return metricsState.accSeries;
    return buildAccuracySeries(36);
  }, [metricsState.accSeries, useMockStream]);

  const f1Full = useMemo(() => {
    if (useMockStream && metricsState.f1Series.length > 0) return metricsState.f1Series;
    return [] as MetricPoint[];
  }, [metricsState.f1Series, useMockStream]);

  const rmseFull = useMemo(() => {
    if (useMockStream && metricsState.rmseSeries.length > 0) return metricsState.rmseSeries;
    return [] as MetricPoint[];
  }, [metricsState.rmseSeries, useMockStream]);

  const [lossVisible, setLossVisible] = useState<LossPoint[]>([]);
  const [accVisible, setAccVisible] = useState<MetricPoint[]>([]);
  const useRmse = metricKind === 'rmse';

  const metricFull = useMemo(() => {
    if (metricKind === 'rmse') return rmseFull;
    if (metricKind === 'f1') return f1Full.length > 0 ? f1Full : accFull;
    return accFull;
  }, [accFull, f1Full, metricKind, rmseFull]);
  const lastLossCountWritten = useRef(0);
  const lastAccCountWritten = useRef(0);
  const lastLossWriteAtRef = useRef(0);
  const lastAccWriteAtRef = useRef(0);
  const lastLossValueRef = useRef<number | null>(null);
  const lastAccValueRef = useRef<number | null>(null);

  const startStep = (nextStepIndex: number, startedAtMs: number) => {
    setStepIndex(nextStepIndex);
    setStepStartedAt(startedAtMs);

    // Reset metric reveal for metric steps.
    setLossVisible([]);
    setAccVisible([]);
    lastLossCountWritten.current = 0;
    lastAccCountWritten.current = 0;
    lastLossWriteAtRef.current = startedAtMs;
    lastAccWriteAtRef.current = startedAtMs;
  };

  useEffect(() => {
    if (!useMockStream) return;
    completedRef.current = false;
    advanceGuardRef.current = -1;
    startStep(0, nowRef.current || performance.now());
  }, [activeScenario, useMockStream]);

  const step = steps[Math.min(stepIndex, steps.length - 1)];
  const elapsed = Math.max(0, now - stepStartedAt);

  const { phaseIndex, phase, phaseProgress } = computeWeightedPhase(step.phases, elapsed, step.durationMs);
  const stepProgress = clamp01(elapsed / Math.max(1, step.durationMs));

  const stepSeed = (seed ?? DEFAULT_SEED) + stepIndex * 1000 + phaseIndex * 100;

  useEffect(() => {
    lastLossValueRef.current = lossVisible.at(-1)?.val_loss ?? null;
  }, [lossVisible]);
  useEffect(() => {
    lastAccValueRef.current = accVisible.at(-1)?.value ?? null;
  }, [accVisible]);

  // RAF clock.
  useEffect(() => {
    let raf = 0;
    let last = 0;

    const tick = (t: number) => {
      raf = requestAnimationFrame(tick);
      nowRef.current = t;

      if (!clockInitRef.current) {
        clockInitRef.current = true;
        last = t;
        setNow(t);
        setStepStartedAt(t);
        return;
      }

      if (t - last > 50) {
        setNow(t);
        last = t;
      }
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Failsafe watchdog per step.
  useEffect(() => {
    if (completedRef.current) return;
    if (stepIndex >= steps.length) return;

    const expected = stepIndex;
    const timeoutMs = Math.max(250, step.durationMs + 400);

    const id = window.setTimeout(() => {
      if (completedRef.current) return;
      if (stepIndexRef.current !== expected) return;
      startStep(expected + 1, nowRef.current || performance.now());
    }, timeoutMs);

    return () => window.clearTimeout(id);
  }, [stepIndex, step.durationMs]);

  // On step start: logs + progress artifact.
  useEffect(() => {
    appendLog(updateFileContentRef.current, `${step.title} — ${step.subtitle}`);
    writeJson(updateFileContentRef.current, '/artifacts/progress.json', {
      step: step.id,
      stepIndex,
      startedAt: new Date().toISOString(),
    });
  }, [step.id, step.title, step.subtitle, stepIndex]);

  // Advance step when complete.
  useEffect(() => {
    if (stepIndex >= steps.length) return;
    if (stepProgress < 1) return;


    if (step.phases.some((p) => p.kind === 'graph')) {
      const graphPhase = step.phases.find((p) => p.kind === 'graph');
      if (graphPhase?.graphType === 'loss' && lossVisible.length < lossFull.length) return;
      if (graphPhase?.graphType === 'accuracy' && accVisible.length < metricFull.length) return;
    }

    if (advanceGuardRef.current === stepIndex) return;
    advanceGuardRef.current = stepIndex;

    if (stepIndex === steps.length - 1) {
      if (completedRef.current) return;
      completedRef.current = true;

      if (!useMockStream) {
        const model = {
          model: 'AutoAI MockNet',
          version: '0.2',
          trainedAt: new Date().toISOString(),
          metrics: {
            loss: lastLossValueRef.current,
            accuracy: lastAccValueRef.current,
          },
        };
        writeJson(updateFileContentRef.current, '/artifacts/model.json', model);
        writeJson(updateFileContentRef.current, '/config/model.json', model);
        appendLog(updateFileContentRef.current, 'Build complete — artifacts written');
      }

      onComplete();
      return;
    }

    const advance = () => {
      startStep(stepIndex + 1, nowRef.current || performance.now());
    };
    if (typeof queueMicrotask === 'function') queueMicrotask(advance);
    else Promise.resolve().then(advance);
  }, [onComplete, stepIndex, stepProgress]);

  // Graph phase metric reveal + artifact writes.
  useEffect(() => {
    if (phase.kind !== 'graph') return;
    const graphType = phase.graphType;

    const late = clamp01((phaseProgress - 0.12) / 0.88);

    if (graphType === 'loss') {
      const targetCount = Math.max(0, Math.floor(lossFull.length * late));
      if (targetCount > lossVisible.length) {
        const next = lossFull.slice(0, targetCount);
        setLossVisible(next);
      }
    }

    if (graphType === 'accuracy') {
      const series = metricFull;
      const targetCount = Math.max(0, Math.floor(series.length * late));
      if (targetCount > accVisible.length) {
        const next = series.slice(0, targetCount);
        setAccVisible(next);
      }
    }
  }, [phase, phaseProgress, lossFull, metricFull, lossVisible.length, accVisible.length]);

  const writeArtifact = (path: string, value: unknown) => {
    writeJson(updateFileContentRef.current, path, value);
  };

  const phaseLabel = getPhaseKindLabel(phase);
  const phaseExtra =
    phase.kind === 'visual'
      ? VISUAL_LABEL[phase.visualId]
      : phase.kind === 'graph'
        ? phase.graphType === 'accuracy'
          ? metricKind
          : phase.graphType
        : '';

  const evaluationMetrics = useMemo(() => {
    const accuracy = metricsState.metricsSummary.accuracy;
    const f1 = metricsState.metricsSummary.f1;
    const rmse = metricsState.metricsSummary.rmse;
    if (useRmse && typeof rmse === 'number') {
      return undefined;
    }
    if (typeof accuracy !== 'number' && typeof f1 !== 'number') return undefined;
    const precision = typeof f1 === 'number' ? Math.min(0.97, f1 + 0.04) : Math.min(0.95, (accuracy ?? 0.8) + 0.03);
    const recall = typeof f1 === 'number' ? Math.max(0.55, f1 - 0.04) : Math.max(0.6, (accuracy ?? 0.8) - 0.04);
    const computedF1 = typeof f1 === 'number' ? f1 : (2 * precision * recall) / Math.max(1e-9, precision + recall);
    const computedAcc = typeof accuracy === 'number' ? accuracy : computedF1;
    return { accuracy: computedAcc, precision, recall, f1: computedF1 };
  }, [metricsState.metricsSummary, useRmse]);

  return (
    <div className="h-full w-full">
      <div className="h-full w-full flex flex-col overflow-hidden">
        {/* Top: step overview grid (no horizontal scroller) */}
        <div className="shrink-0 border-b border-replit-border bg-replit-surface">
          <div className="px-4 pt-4 pb-3">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-lg font-semibold text-replit-text">{step.title}</div>
                <div className="text-sm text-replit-textMuted">{step.subtitle}</div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-xs font-mono text-replit-textMuted whitespace-nowrap">
                  Step {stepIndex + 1}/{steps.length} · {phaseLabel}
                  {phaseExtra ? `: ${phaseExtra}` : ''} · ({phaseIndex + 1}/{step.phases.length})
                </div>
                <label className="text-xs text-replit-textMuted flex items-center gap-2">
                  Scenario
                  <select
                    value={activeScenario}
                    onChange={(e) => setActiveScenario(e.target.value as ScenarioId)}
                    className="rounded-md border border-replit-border bg-replit-bg px-2 py-1 text-xs text-replit-text"
                  >
                    <option value="A">A: Binary LogReg</option>
                    <option value="B">B: Multiclass RF</option>
                    <option value="C">C: Regression Ridge</option>
                  </select>
                </label>
              </div>
            </div>

            <div className="mt-3 py-2">
              <div
                className="grid gap-3"
                style={{ gridTemplateColumns: `repeat(${steps.length}, minmax(0, 1fr))` }}
              >
                {steps.map((s, idx) => {
                  const isActive = idx === stepIndex;
                  const isDone = idx < stepIndex;
                  const nodeBg = isDone
                    ? 'bg-replit-success/80 text-white border-replit-success/80'
                    : isActive
                      ? 'bg-replit-accent/90 text-white border-replit-accent/90'
                      : 'bg-replit-surface/35 text-replit-textMuted border-replit-border/60';

                  return (
                    <div key={s.id} className="flex items-center gap-2">
                      <div
                        className={clsx(
                          'relative w-7 h-7 rounded-full border flex items-center justify-center text-[11px] font-semibold shrink-0',
                          nodeBg
                        )}
                        title={s.title}
                      >
                        {idx + 1}
                        {isActive ? (
                          <div
                            aria-hidden
                            className={clsx(
                              'absolute -inset-1 rounded-full border-2 border-yellow-300/80 border-t-transparent',
                              reducedMotion ? '' : 'animate-spin'
                            )}
                          />
                        ) : null}
                      </div>
                      <div className={clsx('text-[11px] leading-tight', isActive ? 'text-replit-text' : 'text-replit-textMuted')}>
                        {s.title}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Main hero */}
        <div className="flex-1 overflow-hidden relative">
          <div className="p-4 h-full">
            <AnimatePresence mode="wait" initial={false}>
              {/* Operation */}
              {phase.kind === 'operation' ? (
                <motion.div
                  key={`${step.id}-op`}
                  initial={reducedMotion ? { opacity: 1 } : { opacity: 0, y: 12 }}
                  animate={reducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
                  exit={reducedMotion ? { opacity: 1 } : { opacity: 0, y: -12 }}
                  transition={{ duration: reducedMotion ? 0 : 0.35 }}
                  className="h-full"
                >
                  <div className="h-full flex flex-col min-h-0">
                    <div className={clsx(
                      'rounded-2xl border border-replit-border bg-replit-surface shadow-sm p-8 h-full overflow-hidden',
                      step.id === 'matrixOps' && 'pb-[50px]'
                    )}>
                      <div className="text-xs text-replit-textMuted mb-4">Operation</div>

                      <div className="mb-5 rounded-xl border border-replit-border/60 bg-replit-surface/35 p-4">
                        <div className="text-[11px] text-replit-textMuted mb-2">Formula</div>
                        <div className="text-replit-text overflow-hidden">
                          <div className="flex justify-center">
                            <div className="origin-center scale-[1.02] md:scale-[1.08]">
                              <BlockMath math={step.equations[0] ?? ''} />
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="mx-auto max-w-5xl">
                        <MatrixGrid
                          label={step.matrixLabel ?? 'X'}
                          rows={step.matrixRows ?? 6}
                          cols={step.matrixCols ?? 8}
                          timeMs={now}
                          reducedMotion={reducedMotion}
                        />
                      </div>

                      <div className="mt-6 text-xs text-replit-textMuted">
                        {phaseProgress < 0.33
                          ? 'Multiplying (row × column)…'
                          : phaseProgress < 0.66
                            ? 'Accumulating partial sums…'
                            : 'Writing the next tensor…'}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ) : null}

              {/* Graph */}
              {phase.kind === 'graph' ? (
                <motion.div
                  key={`${step.id}-graph-${phase.graphType}`}
                  initial={reducedMotion ? { opacity: 1 } : { opacity: 0, y: 12 }}
                  animate={reducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
                  exit={reducedMotion ? { opacity: 1 } : { opacity: 0, y: -12 }}
                  transition={{ duration: reducedMotion ? 0 : 0.35 }}
                  className="h-full"
                >
                  <div className="h-full flex flex-col min-h-0">
                    <div className="rounded-2xl border border-replit-border bg-replit-surface shadow-sm p-8 h-full overflow-hidden">
                      {phase.graphType === 'loss' ? (
                        <TrainingLossVisualizer
                          data={lossVisible.map((p) => ({ epoch: p.epoch, train_loss: p.train_loss, val_loss: p.val_loss }))}
                        />
                      ) : (
                        <ModelMetricsVisualizer
                          metricKind={metricKind}
                          data={mapMetricSeries(accVisible, metricKind)}
                        />
                      )}
                      <div className="mt-6 text-xs text-replit-textMuted">
                        {phase.graphType === 'loss'
                          ? 'Plotting loss curve…'
                          : useRmse
                            ? 'Plotting RMSE curve…'
                            : 'Plotting accuracy curve…'}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ) : null}

              {/* Visual */}
              {phase.kind === 'visual' ? (
                <motion.div
                  key={`${step.id}-visual-${phase.visualId}`}
                  initial={reducedMotion ? { opacity: 1 } : { opacity: 0, y: 12 }}
                  animate={reducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
                  exit={reducedMotion ? { opacity: 1 } : { opacity: 0, y: -12 }}
                  transition={{ duration: reducedMotion ? 0 : 0.35 }}
                  className="h-full"
                >
                  <div className="h-full flex flex-col min-h-0">
                    <div className={clsx(
                      'rounded-2xl border border-replit-border bg-replit-surface shadow-sm p-8 h-full overflow-hidden',
                      phase.visualId === 'neuralNetForward' && 'pb-[50px]'
                    )}>
                      <div className="text-xs text-replit-textMuted mb-4">{phaseTitle(phase)}</div>
                      {(() => {
                        const C = VISUALS[phase.visualId as VisualId];
                        return (
                          <C
                            timeMs={now}
                            phaseProgress={phaseProgress}
                            seed={stepSeed}
                            reducedMotion={reducedMotion}
                            writeArtifact={phase.visualId === 'evaluation' ? writeArtifact : undefined}
                            confusion={
                              phase.visualId === 'evaluation' && scenarioConfig.showConfusionMatrix
                                ? metricsState.confusionTable ?? undefined
                                : undefined
                            }
                            metrics={phase.visualId === 'evaluation' ? evaluationMetrics : undefined}
                            showConfusion={
                              phase.visualId === 'evaluation'
                                ? scenarioConfig.showConfusionMatrix && (metricsState.confusionTable?.length ?? 0) <= 2
                                : undefined
                            }
                            points={
                              phase.visualId === 'embeddingScatter' && scenarioConfig.showEmbedding
                                ? metricsState.embeddingPoints
                                : undefined
                            }
                            path={phase.visualId === 'gradDescent' ? metricsState.gradientPath : undefined}
                            surfaceSpec={phase.visualId === 'gradDescent' ? metricsState.surfaceSpec : undefined}
                            residuals={phase.visualId === 'residuals' ? metricsState.residuals : undefined}
                          />
                        );
                      })()}
                    </div>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
