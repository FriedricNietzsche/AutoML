import { create } from 'zustand';
import type {
  ClassificationMetricsPayload,
  RegressionMetricsPayload,
  ROCCurveDataPayload,
  PrecisionRecallCurvePayload,
  ConfusionMatrixDataPayload,
  FeatureImportanceDataPayload,
  SHAPExplanationsPayload,
  EvaluationCompletePayload,
  MetricScalarPayload,
} from '../lib/contract';

export type TaskType = 'classification' | 'regression' | null;

export interface MetricHistory {
  name: string;
  values: Array<{ step: number; value: number; split: string }>;
  latest: number;
}

export interface MetricsStoreState {
  // Current run
  currentRunId: string | null;
  taskType: TaskType;
  isEvaluating: boolean;
  
  // Classification metrics
  classificationMetrics: ClassificationMetricsPayload | null;
  
  // Regression metrics
  regressionMetrics: RegressionMetricsPayload | null;
  
  // Curve data
  rocCurve: ROCCurveDataPayload | null;
  prCurve: PrecisionRecallCurvePayload | null;
  
  // Confusion matrix
  confusionMatrix: ConfusionMatrixDataPayload | null;
  
  // Feature importance
  featureImportance: FeatureImportanceDataPayload | null;
  
  // SHAP explanations
  shapExplanations: SHAPExplanationsPayload | null;
  
  // Real-time metric history (during training)
  metricHistory: Record<string, MetricHistory>;
  
  // Final evaluation summary
  evaluationComplete: EvaluationCompletePayload | null;
  
  // Actions
  setRunId: (runId: string) => void;
  setTaskType: (taskType: TaskType) => void;
  setEvaluating: (isEvaluating: boolean) => void;
  updateClassificationMetrics: (metrics: ClassificationMetricsPayload) => void;
  updateRegressionMetrics: (metrics: RegressionMetricsPayload) => void;
  updateROCCurve: (curve: ROCCurveDataPayload) => void;
  updatePRCurve: (curve: PrecisionRecallCurvePayload) => void;
  updateConfusionMatrix: (matrix: ConfusionMatrixDataPayload) => void;
  updateFeatureImportance: (importance: FeatureImportanceDataPayload) => void;
  updateSHAPExplanations: (shap: SHAPExplanationsPayload) => void;
  addMetricScalar: (scalar: MetricScalarPayload) => void;
  setEvaluationComplete: (evaluation: EvaluationCompletePayload) => void;
  reset: () => void;
  
  // Computed getters
  getPrimaryMetric: () => { name: string; value: number } | null;
  hasMetrics: () => boolean;
}

const initialState = {
  currentRunId: null,
  taskType: null,
  isEvaluating: false,
  classificationMetrics: null,
  regressionMetrics: null,
  rocCurve: null,
  prCurve: null,
  confusionMatrix: null,
  featureImportance: null,
  shapExplanations: null,
  metricHistory: {},
  evaluationComplete: null,
};

export const useMetricsStore = create<MetricsStoreState>((set, get) => ({
  ...initialState,
  
  setRunId: (runId: string) => set({ currentRunId: runId }),
  
  setTaskType: (taskType: TaskType) => set({ taskType }),
  
  setEvaluating: (isEvaluating: boolean) => set({ isEvaluating }),
  
  updateClassificationMetrics: (metrics: ClassificationMetricsPayload) => {
    set({ 
      classificationMetrics: metrics,
      taskType: 'classification',
    });
  },
  
  updateRegressionMetrics: (metrics: RegressionMetricsPayload) => {
    set({ 
      regressionMetrics: metrics,
      taskType: 'regression',
    });
  },
  
  updateROCCurve: (curve: ROCCurveDataPayload) => {
    set({ rocCurve: curve });
  },
  
  updatePRCurve: (curve: PrecisionRecallCurvePayload) => {
    set({ prCurve: curve });
  },
  
  updateConfusionMatrix: (matrix: ConfusionMatrixDataPayload) => {
    set({ confusionMatrix: matrix });
  },
  
  updateFeatureImportance: (importance: FeatureImportanceDataPayload) => {
    set({ featureImportance: importance });
  },
  
  updateSHAPExplanations: (shap: SHAPExplanationsPayload) => {
    set({ shapExplanations: shap });
  },
  
  addMetricScalar: (scalar: MetricScalarPayload) => {
    const { metricHistory } = get();
    const key = `${scalar.name}_${scalar.split}`;
    
    const existing = metricHistory[key] || {
      name: scalar.name,
      values: [],
      latest: scalar.value,
    };
    
    const newHistory = {
      ...existing,
      values: [
        ...existing.values,
        { step: scalar.step, value: scalar.value, split: scalar.split },
      ].slice(-100), // Keep last 100 points
      latest: scalar.value,
    };
    
    set({
      metricHistory: {
        ...metricHistory,
        [key]: newHistory,
      },
    });
  },
  
  setEvaluationComplete: (evaluation: EvaluationCompletePayload) => {
    set({ 
      evaluationComplete: evaluation,
      isEvaluating: false,
    });
  },
  
  reset: () => set(initialState),
  
  // Computed getters
  getPrimaryMetric: () => {
    const state = get();
    
    if (state.evaluationComplete) {
      return {
        name: state.evaluationComplete.primary_metric,
        value: state.evaluationComplete.primary_value,
      };
    }
    
    if (state.taskType === 'classification' && state.classificationMetrics) {
      return {
        name: 'Accuracy',
        value: state.classificationMetrics.accuracy,
      };
    }
    
    if (state.taskType === 'regression' && state.regressionMetrics) {
      return {
        name: 'RMSE',
        value: state.regressionMetrics.rmse,
      };
    }
    
    return null;
  },
  
  hasMetrics: () => {
    const state = get();
    return !!(state.classificationMetrics || state.regressionMetrics);
  },
}));
