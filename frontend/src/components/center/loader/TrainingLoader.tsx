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
import type { MockWSEnvelope } from '../../../mock/backendEventTypes';
import type { ArtifactAddedPayload, LogLinePayload } from '../../../lib/contract';
import { SCENARIO_VIZ, type LoaderStepId } from '../../../mock/scenarioVizConfig';
import { useProjectStore } from '../../../store/projectStore';

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

const phaseTitle = (phase: StepPhase) => {
  if (phase.kind === 'operation') return 'Operation';
  if (phase.kind === 'graph') return phase.graphType === 'loss' ? 'Graph (Loss)' : 'Graph (Accuracy)';
  return `Visual (${VISUAL_LABEL[phase.visualId]})`;
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
            {rows}├ù{cols}
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

  // Stage management
  const [currentStage, setCurrentStage] = useState<number>(0); // 0 = initial, 1-5 = stages
  const [isStageRunning, setIsStageRunning] = useState<boolean>(false);
  const [stageCompleted, setStageCompleted] = useState<boolean>(false);
  const [showChangeOption, setShowChangeOption] = useState<boolean>(false);
  const [changeRequest, setChangeRequest] = useState<string>('');

  const [activeScenario, setActiveScenario] = useState<ScenarioId>(scenarioId ?? 'B');
  useEffect(() => {
    if (!scenarioId) return;
    setActiveScenario(scenarioId);
  }, [scenarioId]);

  const { events, metricsState } = useMockAutoMLStream({
    scenarioId: activeScenario,
    seed,
    enabled: useMockStream && currentStage > 0,
  });
  const applyProjectEvent = useProjectStore((state) => state.applyEvent);

  // Load actual image pixels client-side for image data
  const [loadedImagePixels, setLoadedImagePixels] = useState<Array<{ r: number; g: number; b: number }> | null>(null);
  const [imageAnimStartedAt, setImageAnimStartedAt] = useState<number | null>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageOffscreenRef = useRef<HTMLCanvasElement | null>(null);
  const stage1ScrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const isImage = metricsState.datasetPreview?.dataType === 'image';
    const needsClientLoad = !!metricsState.datasetPreview?.imageData?.needsClientLoad;

    if (!isImage || !needsClientLoad) {
      setLoadedImagePixels(null);
      setImageAnimStartedAt(null);
      imageOffscreenRef.current = null;
      return;
    }

    // Only do DOM image extraction when Stage 2 is actually mounted.
    if (currentStage !== 2) return;

    const canvasSize = 280;
    const gridSize = 20;
    let cancelled = false;

    // Use an offscreen canvas for sampling so we don't depend on the on-screen canvas being present.
    const offscreen = document.createElement('canvas');
    offscreen.width = canvasSize;
    offscreen.height = canvasSize;
    const sampleCtx = offscreen.getContext('2d');
    if (!sampleCtx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    let retried = false;

    img.onload = () => {
      if (cancelled) return;

      const scale = Math.min(canvasSize / img.width, canvasSize / img.height);
      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;
      const x = (canvasSize - scaledWidth) / 2;
      const y = (canvasSize - scaledHeight) / 2;

      // Draw to offscreen canvas
      sampleCtx.fillStyle = '#ffffff';
      sampleCtx.fillRect(0, 0, canvasSize, canvasSize);
      sampleCtx.drawImage(img, x, y, scaledWidth, scaledHeight);

      // Keep a handle so we can paint it onto the on-screen canvas later.
      imageOffscreenRef.current = offscreen;

      // Also draw to on-screen canvas if present
      const onCanvas = imageCanvasRef.current;
      const onCtx = onCanvas?.getContext('2d');
      if (onCtx) {
        onCtx.clearRect(0, 0, canvasSize, canvasSize);
        onCtx.drawImage(offscreen, 0, 0);
      }

      // Sample centers of a 20x20 grid from the 280x280 imageData
      const imgData = sampleCtx.getImageData(0, 0, canvasSize, canvasSize);
      const pixels: Array<{ r: number; g: number; b: number }> = [];
      const cellSize = canvasSize / gridSize;

      for (let gy = 0; gy < gridSize; gy++) {
        for (let gx = 0; gx < gridSize; gx++) {
          const centerX = gx * cellSize + cellSize / 2;
          const centerY = gy * cellSize + cellSize / 2;
          const idx = (Math.floor(centerY) * canvasSize + Math.floor(centerX)) * 4;
          pixels.push({ r: imgData.data[idx], g: imgData.data[idx + 1], b: imgData.data[idx + 2] });
        }
      }

      setLoadedImagePixels(pixels);
      setImageAnimStartedAt(nowRef.current || performance.now());
    };

    img.onerror = () => {
      if (cancelled) return;
      if (!retried) {
        retried = true;
        img.src = '/src/assets/image.jpg';
        return;
      }
      console.error('Failed to load image');
    };

    // Use the correct path to the image
    img.src = new URL('../../../assets/image.jpg', import.meta.url).href;
    return () => {
      cancelled = true;
    };
  }, [currentStage, metricsState.datasetPreview?.dataType, metricsState.datasetPreview?.imageData?.needsClientLoad]);

  // If the image finished loading before the on-screen canvas ref was mounted, paint it now.
  useEffect(() => {
    const isImage = metricsState.datasetPreview?.dataType === 'image';
    const needsClientLoad = !!metricsState.datasetPreview?.imageData?.needsClientLoad;
    if (currentStage !== 2 || !isImage || !needsClientLoad) return;
    if (!loadedImagePixels || loadedImagePixels.length === 0) return;

    const canvas = imageCanvasRef.current;
    const offscreen = imageOffscreenRef.current;
    if (!canvas || !offscreen) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0);
  }, [currentStage, loadedImagePixels, metricsState.datasetPreview?.dataType, metricsState.datasetPreview?.imageData?.needsClientLoad]);

  const stage1Thinking = metricsState.thinkingByStage?.DATA_SOURCE ?? [];

  // Keep Stage 1 view pinned to the latest streamed message.
  useEffect(() => {
    if (currentStage !== 1) return;
    const el = stage1ScrollRef.current;
    if (!el) return;

    // Defer until after DOM has painted new content.
    requestAnimationFrame(() => {
      try {
        el.scrollTo({ top: el.scrollHeight, behavior: reducedMotion ? 'auto' : 'smooth' });
      } catch {
        el.scrollTop = el.scrollHeight;
      }
    });
  }, [currentStage, reducedMotion, stage1Thinking.length]);

  const scenarioConfig = useMemo(() => SCENARIO_VIZ[activeScenario], [activeScenario]);
  
  // Define one animation per stage
  const stageDefinitions = useMemo<StepDef[]>(() => [
    {
      id: 'matrixOps' as LoaderStepId,
      title: 'Loading Data',
      subtitle: 'Reading and validating dataset',
      durationMs: 6000,
      phases: [{ kind: 'operation' as const }],
    },
    {
      id: 'preprocessing' as LoaderStepId,
      title: 'Preprocessing',
      subtitle: 'Normalizing and transforming features',
      durationMs: 6000,
      matrixLabel: 'Data Preview',
      matrixRows: 8,
      matrixCols: 6,
      phases: [{ kind: 'operation' as const }],
    },
    {
      id: 'trainLoss' as LoaderStepId,
      title: 'Training Model',
      subtitle: 'Optimizing model parameters',
      durationMs: 8000,
      equations: ['\\mathcal{L}(\\theta)'],
      phases: [{ kind: 'graph' as const, graphType: 'loss' as const }],
    },
    {
      id: 'evaluation' as LoaderStepId,
      title: 'Evaluating Performance',
      subtitle: 'Computing metrics and validation',
      durationMs: 7000,
      equations: ['\\mathrm{F1}=2\\frac{PR}{P+R}'],
      phases: [{ kind: 'visual' as const, visualId: 'evaluation' as const }],
    },
    {
      id: 'embedding' as LoaderStepId,
      title: 'Exporting Model',
      subtitle: 'Packaging for deployment',
      durationMs: 5000,
      equations: ['\\mathbf{z}=f(\\mathbf{x})'],
      phases: [{ kind: 'visual' as const, visualId: 'embeddingScatter' as const }],
    },
  ], []);

  // Get the current stage's step definition
  const steps = useMemo(() => {
    if (currentStage === 0 || currentStage > 5) return stageDefinitions;
    return [stageDefinitions[currentStage - 1]];
  }, [currentStage, stageDefinitions]);

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
      const event = events[i] as MockWSEnvelope;
      applyProjectEvent(event);
      const name = event.event?.name;
      const payload = event.event?.payload as Record<string, unknown> | undefined;

      if (name === 'ARTIFACT_ADDED') {
        const artifactPayload = payload as ArtifactAddedPayload | undefined;
        const artifact = artifactPayload?.artifact;
        const meta = artifact?.meta as Record<string, unknown> | undefined;
        const filePath = meta?.file_path as string | undefined;
        const content = meta?.content as string | undefined;
        if (filePath && typeof content === 'string') {
          updateFileContentRef.current(filePath, content);
        }
      }

      if (name === 'LOG_LINE') {
        const logPayload = payload as LogLinePayload | undefined;
        if (!logPayload?.text) continue;
        appendLog(updateFileContentRef.current, logPayload.text, logPayload.level);
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

  // Handle stage progression
  const handleProceed = () => {
    if (currentStage === 0) {
      // Start stage 1
      setCurrentStage(1);
      setIsStageRunning(true);
      setStageCompleted(false);
      setStepIndex(0);
      setStepStartedAt(nowRef.current || 0);
    } else if (stageCompleted && !isStageRunning) {
      // Move to next stage
      const nextStage = currentStage + 1;
      if (nextStage <= 5) {
        setCurrentStage(nextStage);
        setIsStageRunning(true);
        setStageCompleted(false);
        setShowChangeOption(false);
        setStepIndex(0);
        setStepStartedAt(nowRef.current || 0);
      }
    }
  };

  const handleMakeChanges = () => {
    // Redirect to stage 3 with the change request
    if (changeRequest.trim()) {
      // TODO: Send changeRequest to the bot/backend
      console.log('Change requested:', changeRequest);
      appendLog(updateFileContentRef.current, `User requested changes: ${changeRequest}`);
    }
    setCurrentStage(3);
    setIsStageRunning(true);
    setStageCompleted(false);
    setShowChangeOption(false);
    setChangeRequest('');
    setStepIndex(0);
    setStepStartedAt(nowRef.current || 0);
  };

  const handleDeployment = () => {
    // Proceed to stage 5 (deployment)
    setCurrentStage(5);
    setIsStageRunning(true);
    setStageCompleted(false);
    setShowChangeOption(false);
    setStepIndex(0);
    setStepStartedAt(nowRef.current || 0);
  };

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
  
  const lastLossValueRef = useRef<number | null>(null);
  const lastAccValueRef = useRef<number | null>(null);

  useEffect(() => {
    if (!useMockStream) return;
    completedRef.current = false;
    advanceGuardRef.current = -1;
    setStepIndex(0);
    setStepStartedAt(nowRef.current || 0);
  }, [activeScenario, useMockStream]);

  const step = steps[Math.min(stepIndex, steps.length - 1)];
  const elapsed = Math.max(0, now - stepStartedAt);

  const { phaseIndex, phase, phaseProgress } = computeWeightedPhase(step.phases, elapsed, step.durationMs);

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

  // On step start: logs + progress artifact.
  useEffect(() => {
    if (currentStage === 0 || !isStageRunning) return;
    appendLog(updateFileContentRef.current, `${step.title} ΓÇö ${step.subtitle}`);
    writeJson(updateFileContentRef.current, '/artifacts/progress.json', {
      step: step.id,
      stage: currentStage,
      startedAt: new Date().toISOString(),
    });
  }, [currentStage, isStageRunning, step.id, step.title, step.subtitle]);

  // Reset metric data when starting new stage
  useEffect(() => {
    if (isStageRunning && currentStage > 0) {
      setLossVisible([]);
      setAccVisible([]);
    }
  }, [currentStage, isStageRunning]);

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

  const fixedNodes = [
    { id: 1, label: 'Data Load' },
    { id: 2, label: 'Preprocess' },
    { id: 3, label: 'Train' },
    { id: 4, label: 'Evaluate' },
    { id: 5, label: 'Export' },
  ];

  // Check when stage animation is complete
  useEffect(() => {
    if (!isStageRunning || currentStage === 0) return;
    
    const stageDef = stageDefinitions[currentStage - 1];
    if (!stageDef) return;
    
    const timeout = setTimeout(() => {
      setIsStageRunning(false);
      setStageCompleted(true);
      if (currentStage === 4) {
        setShowChangeOption(true);
      } else if (currentStage === 5) {
        // After deployment, navigate to tester page
        setTimeout(() => {
          onComplete();
        }, 1500); // Give user a moment to see completion message
      }
    }, stageDef.durationMs);
    
    return () => clearTimeout(timeout);
  }, [isStageRunning, currentStage, stageDefinitions, onComplete]);

  const getStagePrompt = () => {
    if (currentStage === 0) {
      return {
        title: "Ready to Build Your AI Model?",
        subtitle: "Click 'Proceed' to start the automated machine learning pipeline. We'll guide you through data loading, preprocessing, training, evaluation, and deployment.",
      };
    }
    if (isStageRunning) {
      return {
        title: step.title,
        subtitle: step.subtitle,
      };
    }
    if (stageCompleted) {
      const stageMessages = [
        { title: "Stage 1 Complete!", subtitle: "Data has been loaded and validated. Ready to proceed to preprocessing?" },
        { title: "Stage 2 Complete!", subtitle: "Data preprocessing finished successfully. Ready to train the model?" },
        { title: "Stage 3 Complete!", subtitle: "Model training completed. Ready to evaluate performance?" },
        { title: "Stage 4 Complete!", subtitle: "Model evaluation finished. Review the results and decide next steps." },
        { title: "Deployment Complete!", subtitle: "Redirecting to model tester..." },
      ];
      return stageMessages[currentStage - 1] || stageMessages[0];
    }
    return { title: "", subtitle: "" };
  };

  const prompt = getStagePrompt();

  return (
    <div className="h-full w-full">
      <div className="h-full w-full flex flex-col overflow-hidden">
        {/* Main hero */}
        <div className="flex-1 overflow-hidden relative">
          <div className="p-4 h-full">
            {/* Header with Proceed button */}
            <div className="mb-4 flex items-center justify-between">
              <div>
                <div className="text-lg font-semibold text-replit-text">{prompt.title}</div>
                <div className="text-sm text-replit-textMuted mt-1">{prompt.subtitle}</div>
              </div>
              <div className="flex items-center gap-3">
                {currentStage === 4 && showChangeOption ? (
                  <>
                    <input
                      type="text"
                      value={changeRequest}
                      onChange={(e) => setChangeRequest(e.target.value)}
                      placeholder="Describe what you'd like to change..."
                      className="px-4 py-2 rounded-lg border border-replit-border bg-replit-bg text-replit-text text-sm placeholder:text-replit-textMuted focus:outline-none focus:ring-2 focus:ring-replit-accent/50 min-w-[300px]"
                    />
                    <button
                      onClick={handleMakeChanges}
                      disabled={!changeRequest.trim()}
                      className={clsx(
                        'px-4 py-2 rounded-lg border text-sm font-medium transition-colors whitespace-nowrap',
                        changeRequest.trim()
                          ? 'border-replit-border bg-replit-surface hover:bg-replit-surfaceHover text-replit-text cursor-pointer'
                          : 'border-replit-border/40 bg-replit-surface/40 text-replit-textMuted cursor-not-allowed'
                      )}
                    >
                      Make Changes
                    </button>
                    <button
                      onClick={handleDeployment}
                      className="px-4 py-2 rounded-lg bg-replit-accent hover:bg-replit-accent/90 text-white text-sm font-medium transition-colors whitespace-nowrap"
                    >
                      Deploy
                    </button>
                  </>
                ) : (
                  <button
                    onClick={handleProceed}
                    disabled={isStageRunning}
                    className={clsx(
                      'px-6 py-2 rounded-lg text-sm font-medium transition-colors',
                      isStageRunning
                        ? 'bg-replit-border/40 text-replit-textMuted cursor-not-allowed'
                        : 'bg-replit-accent hover:bg-replit-accent/90 text-white cursor-pointer'
                    )}
                  >
                    {currentStage === 0 ? 'Start' : isStageRunning ? 'Running...' : 'Proceed'}
                  </button>
                )}
              </div>
            </div>

            {/* Show content only when stage is running */}
            {!isStageRunning && currentStage === 0 && (
              <div className="h-[calc(100%-80px)] flex items-center justify-center">
                <div className="text-center max-w-2xl">
                  <div className="text-6xl mb-6">🚀</div>
                  <h2 className="text-2xl font-bold text-replit-text mb-3">Let's Build Your AI Model</h2>
                  <p className="text-replit-textMuted mb-6">
                    Our automated pipeline will handle everything from data loading to deployment.
                    Click 'Start' when you're ready to begin.
                  </p>
                </div>
              </div>
            )}

            {(isStageRunning || (!isStageRunning && stageCompleted && currentStage > 0)) && (
            <div className="relative h-[calc(100%-80px)]">
              {/* Animation Content - always visible but blurred when completed */}
              <div className={clsx(
                'h-full transition-all duration-500',
                !isStageRunning && stageCompleted && 'blur-sm'
              )}>
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
                    <div className="rounded-2xl border border-replit-border bg-replit-surface shadow-sm p-8 h-full overflow-hidden">
                      {currentStage === 1 ? (
                        <div className="h-full flex flex-col">
                          {/* Thinking Panel */}
                          <div className="flex-1 rounded-xl bg-replit-surface/35 p-6 pb-6 pt-2 min-h-0 flex flex-col">
                            <div className="flex items-center gap-3 mb-3">
                              <div className="text-lg font-semibold text-replit-text">AutoML Assistant</div>
                              {isStageRunning && (
                                <div className={reducedMotion ? 'grid grid-cols-3 gap-0.5 opacity-80' : 'grid grid-cols-3 gap-0.5 opacity-90'}>
                                  {Array.from({ length: 9 }).map((_, i) => {
                                    const colors = ['bg-replit-accent/60', 'bg-replit-success/50', 'bg-replit-warning/50'];
                                    return (
                                      <div
                                        key={i}
                                        className={
                                          `h-2 w-2 rounded-[2px] ${colors[i % colors.length]} ` +
                                          (reducedMotion ? '' : 'animate-[matrixPulse_900ms_ease-in-out_infinite]')
                                        }
                                        style={!reducedMotion ? { animationDelay: `${i * 70}ms` } : undefined}
                                      />
                                    );
                                  })}
                                </div>
                              )}
                            </div>

                            <div
                              ref={stage1ScrollRef}
                              className="flex-1 overflow-auto rounded-lg bg-replit-surface/40 p-6 relative"
                              style={{
                                WebkitMaskImage:
                                  'linear-gradient(to bottom, rgba(0,0,0,0.25) 0%, rgba(0,0,0,0.6) 10%, rgba(0,0,0,1) 32%, rgba(0,0,0,1) 100%)',
                                maskImage:
                                  'linear-gradient(to bottom, rgba(0,0,0,0.25) 0%, rgba(0,0,0,0.6) 10%, rgba(0,0,0,1) 32%, rgba(0,0,0,1) 100%)',
                                WebkitMaskRepeat: 'no-repeat',
                                maskRepeat: 'no-repeat',
                                WebkitMaskSize: '100% 100%',
                                maskSize: '100% 100%',
                              }}
                            >
                              {stage1Thinking.length === 0 ? (
                                <div className="flex items-center gap-3 text-replit-textMuted">
                                  <div className={reducedMotion ? 'grid grid-cols-3 gap-0.5 opacity-80' : 'grid grid-cols-3 gap-0.5 opacity-90'}>
                                    {Array.from({ length: 9 }).map((_, i) => {
                                      const colors = ['bg-replit-accent/50', 'bg-replit-success/40', 'bg-replit-warning/40'];
                                      return (
                                        <div
                                          key={i}
                                          className={
                                            `h-3 w-3 rounded-[2px] ${colors[i % colors.length]} ` +
                                            (reducedMotion ? '' : 'animate-[matrixPulse_900ms_ease-in-out_infinite]')
                                          }
                                          style={!reducedMotion ? { animationDelay: `${i * 70}ms` } : undefined}
                                        />
                                      );
                                    })}
                                  </div>
                                  <span className="text-base">Thinking...</span>
                                </div>
                              ) : (
                                <div className="space-y-4">
                                  {stage1Thinking.slice(-50).map((msg, idx) => (
                                    <div key={`${idx}-${msg.slice(0, 20)}`} className="text-base leading-relaxed text-replit-text">
                                      <span className="text-replit-accent font-medium">▸ </span>
                                      {msg}
                                    </div>
                                  ))}
                                  {isStageRunning && (
                                    <div className="flex items-center gap-2 text-replit-textMuted">
                                      <div className={reducedMotion ? 'grid grid-cols-3 gap-0.5 opacity-80' : 'grid grid-cols-3 gap-0.5 opacity-90'}>
                                        {Array.from({ length: 9 }).map((_, i) => {
                                          const colors = ['bg-replit-accent/50', 'bg-replit-success/40', 'bg-replit-warning/40'];
                                          return (
                                            <div
                                              key={i}
                                              className={
                                                `h-2.5 w-2.5 rounded-[2px] ${colors[i % colors.length]} ` +
                                                (reducedMotion ? '' : 'animate-[matrixPulse_900ms_ease-in-out_infinite]')
                                              }
                                              style={!reducedMotion ? { animationDelay: `${i * 70}ms` } : undefined}
                                            />
                                          );
                                        })}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ) : currentStage === 2 ? (
                        <div className="h-full flex flex-col">
                          {metricsState.datasetPreview?.dataType === 'image' ? (
                            /* Stage 2: Image Vectorization Animation */
                            <div className="flex-1 rounded-xl bg-replit-surface/35 p-6 min-h-0 flex flex-col">
                              <div className="flex items-center justify-between mb-4">
                                <div className="text-base font-semibold text-replit-text">Image Vectorization</div>
                                <div className="text-xs text-replit-textMuted">20×20 pixels → 400-d vector</div>
                              </div>

                              <div className="flex-1 overflow-hidden rounded-lg bg-replit-surface/40 p-4 relative">
                                {(() => {
                                  const totalDuration = step.durationMs;
                                  const imageElapsed = imageAnimStartedAt === null ? 0 : Math.max(0, now - imageAnimStartedAt);
                                  const animStage = imageElapsed < totalDuration * 0.2 ? 0
                                    : imageElapsed < totalDuration * 0.45 ? 1
                                    : imageElapsed < totalDuration * 0.7 ? 2
                                    : 3;

                                  const gridSize = 20;
                                  const canvasSize = 280;
                                  const canvasLeft = 40;
                                  const canvasTop = 56;
                                  const cellSize = canvasSize / gridSize;
                                  const particleSize = 16;

                                  const pixels = loadedImagePixels ?? [];
                                  const hasPixels = pixels.length >= gridSize * gridSize;

                                  const particles = hasPixels
                                    ? pixels.slice(0, gridSize * gridSize).map((pixel, idx) => {
                                        const gridX = idx % gridSize;
                                        const gridY = Math.floor(idx / gridSize);
                                        const startX = canvasLeft + gridX * cellSize + cellSize / 2 - particleSize / 2;
                                        const startY = canvasTop + gridY * cellSize + cellSize / 2 - particleSize / 2;
                                        return { ...pixel, gridX, gridY, startX, startY, idx };
                                      })
                                    : [];

                                  const displayCount = 100;
                                  const displayStart = Math.floor((particles.length - displayCount) / 2);

                                  return (
                                    <>
                                      {/* Original Canvas */}
                                      <div
                                        className="absolute transition-all duration-[1500ms] ease-out"
                                        style={{
                                          left: `${canvasLeft}px`,
                                          top: `${canvasTop}px`,
                                          opacity: !hasPixels ? 0 : animStage >= 1 ? 0 : 1,
                                          transform: animStage >= 1 ? 'scale(0.92)' : 'scale(1)',
                                          filter: animStage >= 1 ? 'blur(10px)' : 'blur(0px)',
                                        }}
                                      >
                                        <canvas
                                          ref={imageCanvasRef}
                                          width={canvasSize}
                                          height={canvasSize}
                                          className="border border-replit-border/70 rounded-xl bg-white/95 shadow-xl"
                                        />
                                        <div className="text-center mt-3 text-replit-textMuted font-medium text-xs">
                                          Original Image
                                        </div>
                                      </div>

                                      {!hasPixels && (
                                        <div className="absolute inset-0 flex items-center justify-center">
                                          <div className="text-sm text-replit-textMuted">Loading image…</div>
                                        </div>
                                      )}

                                      {/* Stage Label */}
                                      {animStage === 1 && (
                                        <motion.div
                                          initial={{ opacity: 0, y: -10 }}
                                          animate={{ opacity: 1, y: 0 }}
                                          className="absolute left-1/2 top-4 transform -translate-x-1/2 text-replit-text text-sm font-semibold"
                                        >
                                          Breaking into pixels...
                                        </motion.div>
                                      )}
                                      {animStage === 2 && (
                                        <motion.div
                                          initial={{ opacity: 0 }}
                                          animate={{ opacity: 1 }}
                                          className="absolute left-64 top-64 text-replit-text text-xs font-medium bg-replit-surface/70 px-2 py-1 rounded"
                                        >
                                          Pixel Grid
                                        </motion.div>
                                      )}
                                      {animStage === 3 && (
                                        <>
                                          <motion.div
                                            initial={{ opacity: 0, y: -10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            className="absolute top-4 left-1/2 transform -translate-x-1/2 text-center"
                                          >
                                            <div className="text-replit-text text-sm font-semibold">Vector Representation</div>
                                            <div className="text-replit-textMuted text-xs">RGB color vectors</div>
                                          </motion.div>
                                          <div className="absolute text-replit-accent text-6xl font-bold left-8 top-1/2 transform -translate-y-1/2">⟨</div>
                                          <div className="absolute text-replit-accent text-6xl font-bold right-8 top-1/2 transform -translate-y-1/2">⟩</div>
                                        </>
                                      )}

                                      {/* Particles */}
                                      {hasPixels &&
                                        particles.map((particle) => {
                                        const getStyle = () => {
                                          const color = `rgb(${particle.r}, ${particle.g}, ${particle.b})`;
                                          const baseTransition = reducedMotion ? 'none' : 'all 1.5s cubic-bezier(0.23, 1, 0.32, 1)';

                                          if (animStage === 0) {
                                            return {
                                              left: `${particle.startX}px`,
                                              top: `${particle.startY}px`,
                                              width: '16px',
                                              height: '16px',
                                              opacity: 0,
                                              transform: 'scale(0)',
                                            };
                                          }

                                          if (animStage === 1) {
                                            return {
                                              left: `${particle.startX}px`,
                                              top: `${particle.startY}px`,
                                              width: '16px',
                                              height: '16px',
                                              backgroundColor: color,
                                              opacity: 1,
                                              transform: 'scale(1)',
                                              transition: baseTransition,
                                              transitionDelay: `${particle.idx * 0.001}s`,
                                            };
                                          }

                                          if (animStage === 2) {
                                            const gridCellSize = 14;
                                            return {
                                              left: `${canvasLeft + 20 + particle.gridX * gridCellSize}px`,
                                              top: `${canvasTop + 30 + particle.gridY * gridCellSize}px`,
                                              width: `${gridCellSize - 1}px`,
                                              height: `${gridCellSize - 1}px`,
                                              backgroundColor: color,
                                              opacity: 1,
                                              transform: 'scale(1)',
                                              transition: baseTransition,
                                              transitionDelay: `${(particle.gridY * gridSize + particle.gridX) * 0.001}s`,
                                            };
                                          }

                                          if (animStage === 3) {
                                            const isVisible = particle.idx >= displayStart && particle.idx < displayStart + displayCount;
                                            if (!isVisible) {
                                              return { opacity: 0, transform: 'scale(0)' };
                                            }

                                            const relativeIdx = particle.idx - displayStart;
                                            const itemsPerColumn = 20;
                                            const columnIdx = Math.floor(relativeIdx / itemsPerColumn);
                                            const rowIdx = relativeIdx % itemsPerColumn;
                                            const vectorX = 100 + columnIdx * 220;
                                            const vectorY = 60 + rowIdx * 28;

                                            return {
                                              left: `${vectorX}px`,
                                              top: `${vectorY}px`,
                                              opacity: 1,
                                              transform: 'scale(1)',
                                              transition: baseTransition,
                                              transitionDelay: `${relativeIdx * 0.0015}s`,
                                            };
                                          }
                                        };

                                        const style = getStyle();
                                        const isVectorStage = animStage === 3;
                                        const isVisible = particle.idx >= displayStart && particle.idx < displayStart + displayCount;

                                        return (
                                          <div
                                            key={particle.idx}
                                            className="absolute will-change-transform"
                                            style={{
                                              ...style,
                                              borderRadius: animStage === 2 ? '1px' : '3px',
                                            }}
                                          >
                                            {isVectorStage && isVisible && (
                                              <div className="flex items-center gap-2 px-3 py-1.5 bg-replit-surface/95 rounded border border-replit-border/60 shadow-lg">
                                                <span className="text-replit-accent font-mono text-[10px] font-semibold min-w-[42px]">
                                                  v[{particle.idx}]
                                                </span>
                                                <span className="text-replit-text font-mono text-[10px] font-bold">
                                                  ({particle.r},{particle.g},{particle.b})
                                                </span>
                                                <div
                                                  className="w-6 h-3 rounded"
                                                  style={{
                                                    backgroundColor: `rgb(${particle.r}, ${particle.g}, ${particle.b})`,
                                                    border: '1px solid rgba(255,255,255,0.2)',
                                                  }}
                                                />
                                              </div>
                                            )}
                                          </div>
                                        );
                                      })}

                                      <div className="absolute bottom-4 left-4 text-xs text-replit-textMuted">
                                        {imageElapsed < totalDuration * 0.2
                                          ? 'Loading image...'
                                          : imageElapsed < totalDuration * 0.45
                                            ? 'Digitizing pixels...'
                                            : imageElapsed < totalDuration * 0.7
                                              ? 'Arranging grid...'
                                              : 'Generating feature vectors...'}
                                      </div>
                                    </>
                                  );
                                })()}
                              </div>
                            </div>
                          ) : (
                            /* Stage 2: Tabular Data Preview with Progressive Row Loading */
                            <div className="flex-1 rounded-xl bg-replit-surface/35 p-6 min-h-0 flex flex-col">
                              <div className="flex items-center justify-between mb-4">
                                <div className="text-base font-semibold text-replit-text">Data Preview</div>
                                <div className="text-xs text-replit-textMuted">70,430 rows × 24 columns</div>
                              </div>

                              <div className="flex-1 overflow-hidden rounded-lg bg-replit-surface/40 p-4 md:p-6 flex flex-col min-h-0">
                                <div className="text-xs text-replit-textMuted mb-3">Profiling schema and computing statistics</div>
                                
                                {/* Matrix Grid with Progressive Row Animation */}
                                <div className="flex-1 min-h-0 flex items-center justify-center">
                                  <div
                                    className="grid gap-px rounded-lg border border-replit-border/60 bg-replit-border/60 p-px overflow-hidden w-full h-full"
                                    style={{ 
                                      gridTemplateColumns: `repeat(${step.matrixCols ?? 6}, minmax(0, 1fr))`,
                                      gridTemplateRows: `repeat(${step.matrixRows ?? 8}, minmax(0, 1fr))`
                                    }}
                                  >
                                    {Array.from({ length: (step.matrixRows ?? 8) * (step.matrixCols ?? 6) }).map((_, idx) => {
                                      const cols = step.matrixCols ?? 6;
                                      const r = Math.floor(idx / cols);
                                      const c = idx % cols;
                                      const previewData = metricsState.datasetPreview;
                                      const display = previewData?.rows?.[r]?.[c] ?? 0;

                                      // Calculate which row should be visible based on time
                                      const totalDuration = step.durationMs;
                                      const rowDuration = totalDuration / (step.matrixRows ?? 8);
                                      const rowStartTime = r * rowDuration;
                                      const rowVisible = elapsed >= rowStartTime;

                                      return (
                                        <motion.div
                                          key={idx}
                                          initial={{ opacity: 0, y: 8 }}
                                          animate={rowVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 8 }}
                                          transition={{
                                            duration: reducedMotion ? 0 : 0.4,
                                            delay: reducedMotion ? 0 : c * 0.05,
                                            ease: 'easeOut'
                                          }}
                                          className="flex items-center justify-center font-mono select-none text-xs md:text-sm bg-replit-surface/40 text-replit-text min-h-0"
                                        >
                                          {typeof display === 'number' ? display.toFixed(2) : display}
                                        </motion.div>
                                      );
                                    })}
                                  </div>
                                </div>

                                <div className="mt-3 text-xs text-replit-textMuted">
                                  {elapsed < step.durationMs * 0.3
                                    ? 'Scanning columns and detecting types...'
                                    : elapsed < step.durationMs * 0.7
                                      ? 'Computing missingness and distributions...'
                                      : 'Finalizing schema profile...'}
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <>
                          <div className="text-xs text-replit-textMuted mb-4">Operation</div>

                          <div className="mb-5 rounded-xl border border-replit-border/60 bg-replit-surface/35 p-4">
                            <div className="text-[11px] text-replit-textMuted mb-2">Formula</div>
                            <div className="text-replit-text overflow-hidden">
                              <div className="flex justify-center">
                                <div className="origin-center scale-[1.02] md:scale-[1.08]">
                                  <BlockMath math={step.equations?.[0] ?? ''} />
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
                              ? 'Multiplying (row ├ù column)ΓÇª'
                              : phaseProgress < 0.66
                                ? 'Accumulating partial sumsΓÇª'
                                : 'Writing the next tensorΓÇª'}
                          </div>
                        </>
                      )}
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
                          ? 'Plotting loss curveΓÇª'
                          : useRmse
                            ? 'Plotting RMSE curveΓÇª'
                            : 'Plotting accuracy curveΓÇª'}
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

            {/* Completion Overlay - shown on top of blurred animation */}
            {!isStageRunning && stageCompleted && currentStage > 0 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
                className="absolute inset-0 flex items-center justify-center pointer-events-none"
              >
                <div className="text-center max-w-2xl bg-replit-surface/95 backdrop-blur-md rounded-2xl border border-replit-border shadow-2xl p-8 pointer-events-auto">
                  <div className="text-6xl mb-6 text-green-500">✓</div>
                  <h2 className="text-2xl font-bold text-replit-text mb-3">
                    {currentStage === 4 && showChangeOption ? 'Ready for Next Steps' : 'Stage Completed!'}
                  </h2>
                  {currentStage === 4 && showChangeOption ? (
                    <p className="text-replit-textMuted mb-6">
                      Would you like to make any changes to your model, or proceed to deployment?
                    </p>
                  ) : (
                    <p className="text-replit-textMuted mb-6">
                      Click 'Proceed' to continue to the next stage.
                    </p>
                  )}
                </div>
              </motion.div>
            )}
            </div>
            )}
          </div>
        </div>

        {/* Bottom: Fixed linked list nodes */}
        <div className="shrink-0 border-t border-replit-border bg-replit-surface">
          <div className="px-4 py-4">
            <div className="flex flex-col items-center gap-2">
              {/* Circles row with connecting lines */}
              <div className="flex items-center justify-center">
                {fixedNodes.map((node, idx) => {
                  const isActive = idx + 1 === currentStage && isStageRunning;
                  const isDone = idx + 1 < currentStage;
                  const nodeBg = isDone
                    ? 'bg-replit-success/80 text-white border-replit-success/80'
                    : isActive
                      ? 'bg-replit-accent/90 text-white border-replit-accent/90'
                      : 'bg-replit-surface/35 text-replit-textMuted border-replit-border/60';

                  return (
                    <div key={node.id} className="flex items-center">
                      <div
                        className={clsx(
                          'relative w-10 h-10 rounded-full border-2 flex items-center justify-center text-sm font-semibold shrink-0 transition-all',
                          nodeBg
                        )}
                      >
                        {node.id}
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
                      {idx < fixedNodes.length - 1 && (
                        <div 
                          className={clsx(
                            'h-1 w-16 -mx-px transition-all duration-500',
                            isDone ? 'bg-replit-success/80' : 'bg-replit-border/60'
                          )}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
              {/* Labels row */}
              <div className="flex items-center justify-center">
                {fixedNodes.map((node, idx) => {
                  const isActive = idx + 1 === currentStage && isStageRunning;
                  return (
                    <div key={node.id} className="flex items-center">
                      <div className={clsx('text-xs font-medium w-10 text-center', isActive ? 'text-replit-text' : 'text-replit-textMuted')}>
                        {node.label}
                      </div>
                      {idx < fixedNodes.length - 1 && (
                        <div className="w-16 -mx-px" />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
