import { useCallback, useEffect, useRef, useState } from 'react';
import type { BackendEvent, StepId } from '../mock/backendEventTypes';
import { createMockAutoMLStream, type MockStreamOptions } from '../mock/mockBackendStream';

export interface BackendDriverState {
  currentStep: StepId;
  logs: string[];
  artifacts: Record<string, string>;
  leaderboard: Array<{ modelName: string; metric: number }>;
  pipelineGraph?: unknown;
  profileSummary?: unknown;
  featureSummary?: unknown;
  reportReady?: boolean;
  exportReady?: boolean;
}

export const initialBackendDriverState: BackendDriverState = {
  currentStep: 'S0',
  logs: [],
  artifacts: {},
  leaderboard: [],
};

export function onBackendEvent(state: BackendDriverState, event: BackendEvent): BackendDriverState {
  switch (event.type) {
    case 'STEP_STATUS':
      return { ...state, currentStep: event.step };
    case 'LOG_LINE':
      return { ...state, logs: [...state.logs, event.message] };
    case 'ARTIFACT_WRITTEN':
      return { ...state, artifacts: { ...state.artifacts, [event.path]: event.content } };
    case 'LEADERBOARD_UPDATED':
      return {
        ...state,
        leaderboard: event.entries.map((entry) => ({
          modelName: entry.model,
          metric: entry.metricValue,
        })),
      };
    case 'PIPELINE_GRAPH':
      return { ...state, pipelineGraph: { nodes: event.nodes, edges: event.edges } };
    case 'PROFILE_SUMMARY':
      return { ...state, profileSummary: event };
    case 'FEATURE_SUMMARY':
      return { ...state, featureSummary: { totalFeatures: event.totalFeatures, topFeatures: event.topFeatures } };
    case 'REPORT_READY':
      return { ...state, reportReady: true };
    case 'EXPORT_READY':
      return { ...state, exportReady: true };
    default:
      return state;
  }
}

export function useMockBackend(options?: MockStreamOptions) {
  const [events, setEvents] = useState<BackendEvent[]>([]);
  const [currentStep, setCurrentStep] = useState<StepId>('S0');
  const [isPaused, setIsPaused] = useState(false);
  const [isRunning, setIsRunning] = useState(false);

  const apiRef = useRef<{
    confirmStep: (stepId: StepId) => void;
    selectPlan: (stepId: StepId, planId: string) => void;
    pause: () => void;
    resume: () => void;
    stop: () => void;
  } | null>(null);
  const runIdRef = useRef(0);

  const start = useCallback((override?: MockStreamOptions) => {
    if (isRunning) return;
    const stream = createMockAutoMLStream({ ...options, ...override });
    apiRef.current = {
      confirmStep: () => undefined,
      selectPlan: () => undefined,
      pause: () => undefined,
      resume: () => undefined,
      stop: () => undefined,
    };
    setIsPaused(false);
    setIsRunning(true);
    setEvents([]);
    runIdRef.current += 1;
    const runId = runIdRef.current;

    (async () => {
      try {
        for await (const event of stream) {
          if (runIdRef.current !== runId) return;
          setEvents((prev) => [...prev, event]);
          if (event.type === 'STEP_STATUS') setCurrentStep(event.step as StepId);
        }
      } finally {
        if (runIdRef.current === runId) setIsRunning(false);
      }
    })();
  }, [isRunning, options]);

  const stop = useCallback(() => {
    apiRef.current?.stop();
    apiRef.current = null;
    setIsRunning(false);
  }, []);

  const confirmStep = useCallback((stepId: StepId) => {
    apiRef.current?.confirmStep(stepId);
  }, []);

  const selectPlan = useCallback((stepId: StepId, planId: string) => {
    apiRef.current?.selectPlan(stepId, planId);
  }, []);

  const pause = useCallback(() => {
    apiRef.current?.pause();
    setIsPaused(true);
  }, []);

  const resume = useCallback(() => {
    apiRef.current?.resume();
    setIsPaused(false);
  }, []);

  useEffect(() => {
    start();
    return () => {
      apiRef.current?.stop();
    };
  }, [start]);

  return {
    events,
    lastEvent: events[events.length - 1],
    currentStep,
    isPaused,
    isRunning,
    start,
    stop,
    confirmStep,
    selectPlan,
    pause,
    resume,
  };
}
