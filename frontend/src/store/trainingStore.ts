import { create } from 'zustand';
import type { TrainProgressPayload, MetricScalarPayload } from '../lib/contract';

// ============================================================================
// TYPES FOR REAL-TIME TRAINING DATA
// ============================================================================

export interface TrainingDataPoint {
  step: number;
  value: number;
  timestamp: number;
}

export interface MetricSeries {
  name: string;
  split: 'train' | 'val' | 'test';
  data: TrainingDataPoint[];
  min: number;
  max: number;
  latest: number;
}

export interface TrainingProgress {
  runId: string;
  epoch: number;
  totalEpochs: number;
  step: number;
  totalSteps: number;
  etaSeconds: number | null;
  phase: string;
  percentComplete: number;
}

export type TrainingStatus = 'idle' | 'initializing' | 'training' | 'evaluating' | 'completed' | 'failed';

// ============================================================================
// TRAINING STORE
// ============================================================================

export interface TrainingStoreState {
  // Status
  status: TrainingStatus;
  currentRunId: string | null;
  startedAt: number | null;
  completedAt: number | null;
  
  // Progress
  progress: TrainingProgress | null;
  
  // Loss and metric curves (for real-time charting)
  lossCurve: MetricSeries | null;
  accuracyCurve: MetricSeries | null;
  metricSeries: Record<string, MetricSeries>;
  
  // Log messages
  logs: Array<{ level: string; text: string; timestamp: number }>;
  
  // Actions
  startTraining: (runId: string) => void;
  updateProgress: (payload: TrainProgressPayload) => void;
  addMetricPoint: (payload: MetricScalarPayload) => void;
  addLog: (level: string, text: string) => void;
  setStatus: (status: TrainingStatus) => void;
  completeTraining: () => void;
  failTraining: (error?: string) => void;
  reset: () => void;
  
  // Getters for charts
  getLossCurveData: () => Array<{ step: number; train?: number; val?: number }>;
  getMetricCurveData: (metricName: string) => Array<{ step: number; train?: number; val?: number; test?: number }>;
  getAllMetricNames: () => string[];
}

const MAX_LOG_ENTRIES = 500;
const MAX_DATA_POINTS = 1000;

const createEmptySeries = (name: string, split: 'train' | 'val' | 'test'): MetricSeries => ({
  name,
  split,
  data: [],
  min: Infinity,
  max: -Infinity,
  latest: 0,
});

export const useTrainingStore = create<TrainingStoreState>((set, get) => ({
  // Initial state
  status: 'idle',
  currentRunId: null,
  startedAt: null,
  completedAt: null,
  progress: null,
  lossCurve: null,
  accuracyCurve: null,
  metricSeries: {},
  logs: [],

  // Actions
  startTraining: (runId: string) => {
    set({
      status: 'initializing',
      currentRunId: runId,
      startedAt: Date.now(),
      completedAt: null,
      progress: null,
      lossCurve: createEmptySeries('loss', 'train'),
      accuracyCurve: null,
      metricSeries: {},
      logs: [],
    });
  },

  updateProgress: (payload: TrainProgressPayload) => {
    const totalProgress = payload.steps > 0 
      ? ((payload.epoch - 1) * payload.steps + payload.step) / (payload.epochs * payload.steps) * 100
      : 0;
    
    set({
      status: 'training',
      progress: {
        runId: payload.run_id,
        epoch: payload.epoch,
        totalEpochs: payload.epochs,
        step: payload.step,
        totalSteps: payload.steps,
        etaSeconds: payload.eta_s,
        phase: payload.phase,
        percentComplete: Math.min(100, totalProgress),
      },
    });
  },

  addMetricPoint: (payload: MetricScalarPayload) => {
    const { name, split, step, value } = payload;
    const seriesKey = `${name}_${split}`;
    const timestamp = Date.now();
    
    set((state) => {
      const existingSeries = state.metricSeries[seriesKey] || createEmptySeries(name, split as 'train' | 'val' | 'test');
      
      // Add new data point
      const newData = [
        ...existingSeries.data,
        { step, value, timestamp },
      ].slice(-MAX_DATA_POINTS); // Keep last N points
      
      const newSeries: MetricSeries = {
        ...existingSeries,
        data: newData,
        min: Math.min(existingSeries.min, value),
        max: Math.max(existingSeries.max, value),
        latest: value,
      };
      
      const updates: Partial<TrainingStoreState> = {
        metricSeries: {
          ...state.metricSeries,
          [seriesKey]: newSeries,
        },
      };
      
      // Also update lossCurve if this is a loss metric
      if (name === 'loss' && split === 'train') {
        updates.lossCurve = newSeries;
      }
      
      // Also update accuracyCurve if this is accuracy
      if (name === 'accuracy') {
        updates.accuracyCurve = newSeries;
      }
      
      return updates;
    });
  },

  addLog: (level: string, text: string) => {
    set((state) => ({
      logs: [
        ...state.logs,
        { level, text, timestamp: Date.now() },
      ].slice(-MAX_LOG_ENTRIES),
    }));
  },

  setStatus: (status: TrainingStatus) => set({ status }),

  completeTraining: () => {
    set({
      status: 'completed',
      completedAt: Date.now(),
    });
  },

  failTraining: (error?: string) => {
    set((state) => ({
      status: 'failed',
      completedAt: Date.now(),
      logs: error 
        ? [...state.logs, { level: 'ERROR', text: error, timestamp: Date.now() }]
        : state.logs,
    }));
  },

  reset: () => {
    set({
      status: 'idle',
      currentRunId: null,
      startedAt: null,
      completedAt: null,
      progress: null,
      lossCurve: null,
      accuracyCurve: null,
      metricSeries: {},
      logs: [],
    });
  },

  // Getters for chart data (formatted for easy charting)
  getLossCurveData: () => {
    const state = get();
    const trainData = state.metricSeries['loss_train']?.data || [];
    const valData = state.metricSeries['loss_val']?.data || [];
    
    // Merge train and val data by step
    const stepMap = new Map<number, { step: number; train?: number; val?: number }>();
    
    trainData.forEach(point => {
      const existing = stepMap.get(point.step) || { step: point.step };
      existing.train = point.value;
      stepMap.set(point.step, existing);
    });
    
    valData.forEach(point => {
      const existing = stepMap.get(point.step) || { step: point.step };
      existing.val = point.value;
      stepMap.set(point.step, existing);
    });
    
    return Array.from(stepMap.values()).sort((a, b) => a.step - b.step);
  },

  getMetricCurveData: (metricName: string) => {
    const state = get();
    const trainData = state.metricSeries[`${metricName}_train`]?.data || [];
    const valData = state.metricSeries[`${metricName}_val`]?.data || [];
    const testData = state.metricSeries[`${metricName}_test`]?.data || [];
    
    const stepMap = new Map<number, { step: number; train?: number; val?: number; test?: number }>();
    
    trainData.forEach(point => {
      const existing = stepMap.get(point.step) || { step: point.step };
      existing.train = point.value;
      stepMap.set(point.step, existing);
    });
    
    valData.forEach(point => {
      const existing = stepMap.get(point.step) || { step: point.step };
      existing.val = point.value;
      stepMap.set(point.step, existing);
    });
    
    testData.forEach(point => {
      const existing = stepMap.get(point.step) || { step: point.step };
      existing.test = point.value;
      stepMap.set(point.step, existing);
    });
    
    return Array.from(stepMap.values()).sort((a, b) => a.step - b.step);
  },

  getAllMetricNames: () => {
    const state = get();
    const names = new Set<string>();
    Object.keys(state.metricSeries).forEach(key => {
      const [name] = key.split('_');
      names.add(name);
    });
    return Array.from(names);
  },
}));

// ============================================================================
// HOOKS FOR EASY ACCESS
// ============================================================================

export function useTrainingProgress() {
  return useTrainingStore((state) => ({
    status: state.status,
    progress: state.progress,
    isTraining: state.status === 'training' || state.status === 'initializing',
    isComplete: state.status === 'completed',
    percentComplete: state.progress?.percentComplete ?? 0,
  }));
}

export function useLossCurve() {
  return useTrainingStore((state) => ({
    data: state.getLossCurveData(),
    latest: state.lossCurve?.latest ?? null,
    min: state.lossCurve?.min ?? null,
    max: state.lossCurve?.max ?? null,
    hasData: (state.lossCurve?.data.length ?? 0) > 0,
  }));
}

export function useTrainingLogs() {
  return useTrainingStore((state) => state.logs);
}
