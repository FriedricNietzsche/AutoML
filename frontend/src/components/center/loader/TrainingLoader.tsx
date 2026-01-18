import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import 'katex/dist/katex.min.css';
import clsx from 'clsx';

import type { StepPhase, VisualId } from './types';
import { clamp01 } from './types';
import { VISUAL_LABEL, VISUALS } from './visuals/visualRegistry';
import TrainingLossVisualizer from './TrainingLossVisualizer';
import ModelMetricsVisualizer from './ModelMetricsVisualizer';
import { Stage1DataLoading } from './stages/Stage1DataLoading';
import { Stage2Preprocessing } from './stages/Stage2Preprocessing';
import { Stage3Training } from './stages/Stage3Training';
import { Stage4Evaluation } from './stages/Stage4Evaluation';
import { FIXED_NODES, STAGE_DEFINITIONS, getStagePrompt } from './utils/stageDefinitions';
import { useMockAutoMLStream } from '../../../mock/useMockAutoMLStream';
import { useLiveMetrics } from '../../../hooks/useLiveMetrics';
import type { ScenarioId } from '../../../mock/scenarios';
import { SCENARIO_VIZ, type LoaderStepId } from '../../../mock/scenarioVizConfig';
import { useProjectStore } from '../../../store/projectStore';

// Utilities
import { 
  mapMetricSeries, 
  computeWeightedPhase, 
  writeJson, 
  appendLog, 
  buildLossSeries,
  buildAccuracySeries,
  type LossPoint, 
  type MetricPoint 
} from './utils/loaderHelpers';

// Custom hooks
import { useLoaderClock, useEventProcessor, useImageVectorization, useMetricsData, useStageManager } from './hooks';

export interface TrainingLoaderProps {
  onComplete: () => void;
  updateFileContent: (path: string, content: string | ((prev: string) => string)) => void;
  scenarioId?: ScenarioId;
  seed?: number;
  useMockStream?: boolean;
  onStart?: () => void | Promise<void>;
}

const DEFAULT_SEED = 1337;

const phaseTitle = (phase: StepPhase) => {
  if (phase.kind === 'operation') return 'Operation';
  if (phase.kind === 'graph') return phase.graphType === 'loss' ? 'Graph (Loss)' : 'Graph (Accuracy)';
  return `Visual (${VISUAL_LABEL[phase.visualId]})`;
};

export default function TrainingLoader({ onComplete, updateFileContent, scenarioId, seed, useMockStream = false, onStart }: TrainingLoaderProps) {
  const reducedMotionPref = useReducedMotion();
  const reducedMotion = !!reducedMotionPref;

  const { now, nowRef } = useLoaderClock();

  // 1. Get Live Data from Store
  const { events: liveEvents, currentStageId, stages } = useProjectStore((state) => ({ 
    events: state.events, 
    currentStageId: state.currentStageId,
    stages: state.stages,
  }));
  const { metricsState: liveMetrics } = useLiveMetrics();

  // 2. Manage Stage Progression (Sync with Backend or Mock Timer)
  const {
    currentStage,
    isStageRunning,
    stageCompleted,
    showChangeOption,
    changeRequest,
    stepIndex,
    stepStartedAt,
    stepIndexRef,
    setChangeRequest,
    handleProceed,
    handleMakeChanges,
    handleDeployment,
    setStepIndex,
    setStepStartedAt,
  } = useStageManager({
    onComplete,
    updateFileContent,
    nowRef,
    backendStageId: currentStageId,
    backendStageStatus: currentStageId ? stages[currentStageId]?.status : undefined,
    useMockStream,
    onStart,
  });

  const [activeScenario, setActiveScenario] = useState<ScenarioId>(scenarioId ?? 'B');
  useEffect(() => {
    if (!scenarioId) return;
    setActiveScenario(scenarioId);
  }, [scenarioId]);

  // 3. Get Mock Data (if enabled)
  const { events: mockEvents, metricsState: mockMetrics } = useMockAutoMLStream({
    scenarioId: activeScenario,
    seed,
    enabled: useMockStream && currentStage > 0,
  });
  
  const metricsState = useMockStream ? mockMetrics : liveMetrics;
  const events = useMockStream ? mockEvents : liveEvents;
  const applyProjectEvent = useProjectStore((state) => state.applyEvent);

  const stage1Thinking = metricsState.thinkingByStage?.DATA_SOURCE ?? [];
  const stage1ScrollRef = useRef<HTMLDivElement | null>(null);

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

  // Get the current stage's step definition
  const steps = useMemo(() => {
    if (currentStage === 0 || currentStage > STAGE_DEFINITIONS.length) return STAGE_DEFINITIONS;
    return [STAGE_DEFINITIONS[currentStage - 1]];
  }, [currentStage]);

  const metricKind = scenarioConfig.metricKind;

  // Use event processor hook
  useEventProcessor({
    events,
    useMockStream,
    updateFileContent,
    applyProjectEvent,
  });

  // Use image vectorization hook  
  const {
    imageAnimStartedAt,
    imageCanvasRef,
    imageOffscreenRef,
  } = useImageVectorization({
    currentStage,
    dataType: metricsState.datasetPreview?.dataType,
    needsClientLoad: metricsState.datasetPreview?.imageData?.needsClientLoad,
    nowRef,
  });

  const completedRef = useRef(false);
  const advanceGuardRef = useRef(-1);
  const lastLossValueRef = useRef<number | null>(null);
  const lastAccValueRef = useRef<number | null>(null);

  useEffect(() => {
    if (!useMockStream) return;
    completedRef.current = false;
    advanceGuardRef.current = -1;
    setStepIndex(0);
    setStepStartedAt(nowRef.current || 0);
  }, [activeScenario, useMockStream, nowRef, setStepIndex, setStepStartedAt]);

  const step = steps[Math.min(stepIndex, steps.length - 1)];
  const elapsed = Math.max(0, now - stepStartedAt);

  const { phaseIndex, phase, phaseProgress } = computeWeightedPhase(step.phases, elapsed, step.durationMs);

  const stepSeed = (seed ?? DEFAULT_SEED) + stepIndex * 1000 + phaseIndex * 100;

  // Use metrics data hook (after phase is computed)
  const {
    lossVisible,
    accVisible,
    lossFull,
    accFull,
    metricFull,
    setLossVisible,
    setAccVisible,
  } = useMetricsData({
    metricsState,
    useMockStream,
    metricKind,
    phase,
    phaseProgress,
  });

  const useRmse = metricKind === 'rmse';

  useEffect(() => {
    lastLossValueRef.current = lossVisible.at(-1)?.val_loss ?? null;
  }, [lossVisible]);
  useEffect(() => {
    lastAccValueRef.current = accVisible.at(-1)?.value ?? null;
  }, [accVisible]);

  // On step start: logs + progress artifact.
  useEffect(() => {
    if (currentStage === 0 || !isStageRunning) return;
    appendLog(updateFileContent, `${step.title} — ${step.subtitle}`);
    writeJson(updateFileContent, '/artifacts/progress.json', {
      step: step.id,
      stage: currentStage,
      startedAt: new Date().toISOString(),
    });
  }, [currentStage, isStageRunning, step.id, step.title, step.subtitle, updateFileContent]);

  // Auto-start and Auto-proceed
  useEffect(() => {
    if (currentStage === 0 && !isStageRunning && onStart) {
      onStart();
    } else if (stageCompleted && !isStageRunning && currentStage < 4 && currentStage > 0) {
      // Auto-proceed for intermediate stages
       const t = setTimeout(() => {
         handleProceed();
       }, 1000);
       return () => clearTimeout(t);
    }
  }, [currentStage, isStageRunning, stageCompleted, onStart, handleProceed]);

  const writeArtifact = (path: string, value: unknown) => {
    writeJson(updateFileContent, path, value);
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
    ...FIXED_NODES,
  ];

  const prompt = getStagePrompt(currentStage, isStageRunning, stageCompleted, step.title, step.subtitle);

  return (
    <div className="h-full w-full">
      <div className="h-full w-full flex flex-col overflow-hidden">
        {/* Main hero */}
        <div className="flex-1 overflow-hidden relative">
          <div className="p-4 h-full">
            {/* Header with Proceed button */}
            <div className="mb-4 flex items-center justify-between">
              {(currentStage > 0 || isStageRunning) && (
                <div>
                  <div className="text-lg font-semibold text-replit-text">{prompt.title}</div>
                  <div className="text-sm text-replit-textMuted mt-1">{prompt.subtitle}</div>
                </div>
              )}
              <div className={clsx("flex items-center gap-3", currentStage === 0 && !isStageRunning && "ml-auto")}>
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
                ) : null}
              </div>
            </div>

            {/* Show content only when stage is running */}
            {!isStageRunning && currentStage === 0 && (
              <div className="h-[calc(100%-80px)] flex items-center justify-center">
                <motion.div 
                  initial={reducedMotion ? { opacity: 1 } : { opacity: 0, scale: 0.95 }}
                  animate={reducedMotion ? { opacity: 1 } : { opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className="text-center max-w-3xl"
                >
                  {/* Rocket emoji with gradient background */}
                  <div className="relative inline-block mb-8">
                    <div className="absolute inset-0 bg-gradient-to-br from-replit-accent/20 via-purple-500/20 to-pink-500/20 rounded-full blur-3xl" />
                    <motion.div 
                      className="relative text-8xl"
                      animate={reducedMotion ? {} : { 
                        y: [0, -8, 0],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    >
                      🚀
                    </motion.div>
                  </div>
                  
                  {/* Title with gradient text */}
                  <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-replit-text via-replit-accent to-purple-400 bg-clip-text text-transparent">
                    Let's Build Your AI Model
                  </h2>
                  
                  {/* Subtitle */}
                  <p className="text-lg text-replit-textMuted mb-8 max-w-xl mx-auto leading-relaxed">
                    Our automated pipeline will handle everything from data loading to deployment.
                    Click <span className="font-semibold text-replit-text">Start</span> when you're ready to begin.
                  </p>
                  
                  {/* Feature pills */}
                  <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
                    {['Smart Data Processing', 'Auto Model Selection', 'One-Click Deploy'].map((feature, idx) => (
                      <motion.div
                        key={feature}
                        initial={reducedMotion ? { opacity: 1 } : { opacity: 0, y: 10 }}
                        animate={reducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.1 + 0.3, duration: 0.3 }}
                        className="px-4 py-2 rounded-full bg-replit-surface border border-replit-border/60 text-sm text-replit-textMuted backdrop-blur-sm"
                      >
                        {feature}
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
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
                      {/* Stage 1: Data Loading */}
            {currentStage - 1 === 0 && (
              <Stage1DataLoading
                metricsState={metricsState}
                stage1ScrollRef={stage1ScrollRef}
                isStageRunning={isStageRunning}
                reducedMotion={reducedMotion}
              />
            )}

            {/* Stage 2: Preprocessing */}
            {currentStage - 1 === 1 && (
              <Stage2Preprocessing
                metricsState={metricsState}
                step={step}
                now={now}
                imageAnimStartedAt={imageAnimStartedAt}
                imageCanvasRef={imageCanvasRef}
                reducedMotion={reducedMotion}
                stepSeed={stepSeed}
              />
            )}

            {/* Stage 3: Training */}
            {currentStage - 1 === 2 && (
              <Stage3Training metricsState={metricsState} />
            )}

            {/* Stage 4: Evaluation */}
            {currentStage - 1 === 3 && (
              <Stage4Evaluation
                metricsState={metricsState}
                metricKind={metricKind}
              />
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
                                : phase.visualId === 'confusionMatrix'
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
