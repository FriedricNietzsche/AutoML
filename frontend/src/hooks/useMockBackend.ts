import { useCallback, useEffect, useRef, useState } from 'react';
import type { MockWSEnvelope } from '../mock/backendEventTypes';
import type { StageID } from '../lib/contract';
import { createMockAutoMLStream, type MockStreamOptions } from '../mock/mockBackendStream';

export interface BackendDriverState {
  currentStage: StageID;
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
  currentStage: 'PARSE_INTENT',
  logs: [],
  artifacts: {},
  leaderboard: [],
};

export function onBackendEvent(state: BackendDriverState, event: MockWSEnvelope): BackendDriverState {
  const name = event.event?.name;
  const payload = event.event?.payload as Record<string, unknown> | undefined;
  if (name === 'STAGE_STATUS') {
    const stageId = payload?.stage_id as StageID | undefined;
    if (stageId) return { ...state, currentStage: stageId };
  }
  if (name === 'LOG_LINE') {
    const text = payload?.text as string | undefined;
    if (text) return { ...state, logs: [...state.logs, text] };
  }
  if (name === 'ARTIFACT_ADDED') {
    const artifact = payload?.artifact as { meta?: Record<string, unknown> } | undefined;
    const meta = artifact?.meta;
    const filePath = meta?.file_path as string | undefined;
    const content = meta?.content as string | undefined;
    if (filePath && typeof content === 'string') {
      return { ...state, artifacts: { ...state.artifacts, [filePath]: content } };
    }
  }
  if (name === 'LEADERBOARD_UPDATED') {
    const rows = (payload?.rows as Array<{ model: string; metric: number }>) ?? [];
    return {
      ...state,
      leaderboard: rows.map((row) => ({ modelName: row.model, metric: row.metric })),
    };
  }
  if (name === 'REPORT_READY') return { ...state, reportReady: true };
  if (name === 'EXPORT_READY') return { ...state, exportReady: true };
  return state;
}

export function useMockBackend(options?: MockStreamOptions) {
  const [events, setEvents] = useState<MockWSEnvelope[]>([]);
  const [currentStage, setCurrentStage] = useState<StageID>('PARSE_INTENT');
  const [isPaused, setIsPaused] = useState(false);
  const [isRunning, setIsRunning] = useState(false);

  const apiRef = useRef<{
    confirmStep: (_stepId: StageID) => void;
    selectPlan: (_stepId: StageID, _planId: string) => void;
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
          const stageId = event.event?.payload?.stage_id as StageID | undefined;
          if (event.event?.name === 'STAGE_STATUS' && stageId) setCurrentStage(stageId);
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

  const confirmStep = useCallback((stepId: StageID) => {
    apiRef.current?.confirmStep(stepId);
  }, []);

  const selectPlan = useCallback((stepId: StageID, planId: string) => {
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
    currentStage,
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
