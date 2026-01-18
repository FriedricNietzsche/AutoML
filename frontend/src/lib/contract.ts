export type StageID =
  | 'PARSE_INTENT'
  | 'DATA_SOURCE'
  | 'PROFILE_DATA'
  | 'PREPROCESS'
  | 'MODEL_SELECT'
  | 'TRAIN'
  | 'REVIEW_EDIT'
  | 'EXPORT';

export type StageStatus = 'PENDING' | 'IN_PROGRESS' | 'WAITING_CONFIRMATION' | 'COMPLETED' | 'FAILED' | 'SKIPPED';

export type EventType =
  | 'HELLO'
  | 'STAGE_STATUS'
  | 'WAITING_CONFIRMATION'
  | 'PLAN_PROPOSED'
  | 'PLAN_APPROVED'
  | 'FILE_TREE_UPDATE'
  | 'ARTIFACT_ADDED'
  | 'PROMPT_PARSED'
  | 'DATASET_CANDIDATES'
  | 'DATASET_SELECTED'
  | 'MODEL_CANDIDATES'
  | 'MODEL_SELECTED'
  | 'DATASET_SAMPLE_READY'
  | 'PROFILE_PROGRESS'
  | 'PROFILE_SUMMARY'
  | 'MISSINGNESS_TABLE_READY'
  | 'TARGET_DISTRIBUTION_READY'
  | 'SPLIT_SUMMARY'
  | 'PREPROCESS_PLAN'
  | 'TRAIN_RUN_STARTED'
  | 'TRAIN_PROGRESS'
  | 'METRIC_SCALAR'
  | 'LEADERBOARD_UPDATED'
  | 'BEST_MODEL_UPDATED'
  | 'CONFUSION_MATRIX_READY'
  | 'ROC_CURVE_READY'
  | 'RESIDUALS_PLOT_READY'
  | 'FEATURE_IMPORTANCE_READY'
  | 'RESOURCE_STATS'
  | 'LOG_LINE'
  | 'TRAIN_RUN_FINISHED'
  | 'EVALUATION_STARTED'
  | 'EVALUATION_METRICS_READY'
  | 'CLASSIFICATION_METRICS_READY'
  | 'REGRESSION_METRICS_READY'
  | 'PRECISION_RECALL_CURVE_READY'
  | 'SHAP_EXPLANATIONS_READY'
  | 'EVALUATION_COMPLETE'
  | 'REPORT_READY'
  | 'NOTEBOOK_READY'
  | 'CODE_WORKSPACE_READY'
  | 'EDIT_SUGGESTIONS'
  | 'EXPORT_PROGRESS'
  | 'EXPORT_READY';

export interface StageDefinition {
  id: StageID;
  label: string;
  description: string;
}

export const STAGE_ORDER: StageDefinition[] = [
  { id: 'PARSE_INTENT', label: 'Parse Intent', description: 'Understand user goal + constraints' },
  { id: 'DATA_SOURCE', label: 'Data Source', description: 'Ingest dataset + sample rows' },
  { id: 'PROFILE_DATA', label: 'Profile Data', description: 'Profile columns + quality' },
  { id: 'PREPROCESS', label: 'Preprocess', description: 'Plan preprocessing steps' },
  { id: 'MODEL_SELECT', label: 'Model Select', description: 'Pick candidate models' },
  { id: 'TRAIN', label: 'Train', description: 'Train + stream metrics' },
  { id: 'REVIEW_EDIT', label: 'Review', description: 'Review artifacts + edits' },
  { id: 'EXPORT', label: 'Export', description: 'Package notebook + model' },
];

export const stageIndex = (stageId: StageID) => STAGE_ORDER.findIndex((s) => s.id === stageId);

export interface StageStatusPayload {
  stage_id: StageID;
  status: StageStatus;
  message?: string;
}

export interface WaitingConfirmationPayload {
  stage_id: StageID;
  summary: string;
  next_actions: string[];
}

export interface StageInfo {
  id: StageID;
  index: number;
  status: StageStatus;
}

export interface WSEventPayload<T extends EventType = EventType, P = unknown> {
  name: T;
  payload: P;
}

export interface WSEnvelope<T extends EventType = EventType, P = unknown> {
  v: number;
  type: 'EVENT';
  project_id: string;
  seq: number;
  ts: number;
  stage: StageInfo;
  event: WSEventPayload<T, P>;
}

export interface LogLinePayload {
  run_id: string;
  level: 'INFO' | 'WARN' | 'ERROR';
  text: string;
}

export interface ProfileProgressPayload {
  phase: string;
  pct: number;
}

export interface DatasetSampleReadyPayload {
  asset_url: string;
  columns: string[];
  n_rows: number;
}

export interface ProfileSummaryPayload {
  n_rows: number;
  n_cols: number;
  missing_pct: number;
  types_breakdown: Record<string, number>;
  warnings: string[];
}

export interface TrainProgressPayload {
  run_id: string;
  epoch: number;
  epochs: number;
  step: number;
  steps: number;
  eta_s: number | null;
  phase: 'init' | 'fit' | 'eval' | 'finalize' | string;
}

export interface MetricScalarPayload {
  run_id: string;
  name: string;
  split: 'train' | 'val' | 'test';
  step: number;
  value: number;
}

export interface ArtifactInfo {
  id: string;
  type: string;
  name: string;
  url: string;
  meta?: Record<string, unknown>;
}

export interface ArtifactAddedPayload {
  artifact: ArtifactInfo;
}

// ============================================================================
// EVALUATION METRICS PAYLOADS
// ============================================================================

export interface EvaluationStartedPayload {
  run_id: string;
  task_type: 'classification' | 'regression';
  message: string;
}

export interface ClassificationMetricsPayload {
  run_id: string;
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc?: number;
  mcc: number; // Matthews Correlation Coefficient
  cohen_kappa: number;
  log_loss?: number;
  average_precision?: number;
  specificity?: number;
  sensitivity?: number;
  n_classes: number;
  class_labels: unknown[];
  class_distribution: Record<string, number>;
  precision_per_class?: number[];
  recall_per_class?: number[];
  f1_per_class?: number[];
}

export interface RegressionMetricsPayload {
  run_id: string;
  mse: number;
  rmse: number;
  mae: number;
  median_ae: number;
  r2: number;
  explained_variance: number;
  max_error: number;
  mape?: number;
  smape?: number;
  n_samples: number;
}

export interface ROCCurveDataPayload {
  run_id: string;
  fpr: number[];
  tpr: number[];
  thresholds: number[];
  auc: number;
  asset_url?: string;
}

export interface PrecisionRecallCurvePayload {
  run_id: string;
  precision: number[];
  recall: number[];
  thresholds: number[];
  average_precision: number;
  asset_url?: string;
}

export interface ConfusionMatrixDataPayload {
  run_id: string;
  matrix: number[][];
  labels?: string[];
  true_positives?: number;
  true_negatives?: number;
  false_positives?: number;
  false_negatives?: number;
  asset_url?: string;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface FeatureImportanceDataPayload {
  run_id: string;
  features: FeatureImportanceItem[];
  method: string;
  asset_url?: string;
}

export interface SHAPExplanationsPayload {
  run_id: string;
  available: boolean;
  feature_names?: string[];
  global_importance?: Record<string, number>;
  importance_ranking?: FeatureImportanceItem[];
  message?: string;
  asset_url?: string;
}

export interface EvaluationCompletePayload {
  run_id: string;
  task_type: 'classification' | 'regression';
  primary_metric: string;
  primary_value: number;
  all_metrics: Record<string, unknown>;
  artifacts: Array<{ type: string; url: string }>;
  shap_available: boolean;
}
