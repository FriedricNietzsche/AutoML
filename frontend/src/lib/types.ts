import { z } from "zod";

// ============================================================================
// STAGE DEFINITIONS
// ============================================================================

export const StageIDSchema = z.enum([
  "PARSE_INTENT",
  "DATA_SOURCE",
  "PROFILE_DATA",
  "PREPROCESS",
  "MODEL_SELECT",
  "TRAIN",
  "REVIEW_EDIT",
  "EXPORT",
]);

export type StageID = z.infer<typeof StageIDSchema>;

export const StageStatusSchema = z.enum([
  "PENDING",
  "IN_PROGRESS",
  "WAITING_CONFIRMATION",
  "COMPLETED",
  "FAILED",
  "SKIPPED",
]);

export type StageStatus = z.infer<typeof StageStatusSchema>;

// ============================================================================
// EVENT TYPE DEFINITIONS
// ============================================================================

export const EventTypeSchema = z.enum([
  // Connection lifecycle
  "HELLO",

  // Global Events
  "STAGE_STATUS",
  "WAITING_CONFIRMATION",
  "PLAN_PROPOSED",
  "PLAN_APPROVED",
  "FILE_TREE_UPDATE",
  "ARTIFACT_ADDED",

  // Stage 1: DATA COLLECTION / MODEL CHOICE
  "PROMPT_PARSED",
  "DATASET_CANDIDATES",
  "DATASET_SELECTED",
  "MODEL_CANDIDATES",
  "MODEL_SELECTED",
  "DATASET_SAMPLE_READY",

  // Stage 2: PROFILING / PREPROCESSING
  "PROFILE_PROGRESS",
  "PROFILE_SUMMARY",
  "MISSINGNESS_TABLE_READY",
  "TARGET_DISTRIBUTION_READY",
  "SPLIT_SUMMARY",
  "PREPROCESS_PLAN",

  // Stage 3: TRAINING (RICH)
  "TRAIN_RUN_STARTED",
  "TRAIN_PROGRESS",
  "METRIC_SCALAR",
  "LEADERBOARD_UPDATED",
  "BEST_MODEL_UPDATED",
  "CONFUSION_MATRIX_READY",
  "ROC_CURVE_READY",
  "RESIDUALS_PLOT_READY",
  "FEATURE_IMPORTANCE_READY",
  "RESOURCE_STATS",
  "LOG_LINE",
  "TRAIN_RUN_FINISHED",

  // Stage 4: REVIEW / EDIT
  "REPORT_READY",
  "NOTEBOOK_READY",
  "CODE_WORKSPACE_READY",
  "EDIT_SUGGESTIONS",

  // Stage 5: EXPORT
  "EXPORT_PROGRESS",
  "EXPORT_READY",
]);

export type EventType = z.infer<typeof EventTypeSchema>;

// ============================================================================
// WEBSOCKET ENVELOPE & CORE MODELS
// ============================================================================

export const StageSchema = z.object({
  id: StageIDSchema,
  index: z.number(),
  status: StageStatusSchema,
});

export type Stage = z.infer<typeof StageSchema>;

export const EventPayloadSchema = z.object({
  name: EventTypeSchema,
  payload: z.record(z.string(), z.any()),
});

export type EventPayload = z.infer<typeof EventPayloadSchema>;

export const EventMessageSchema = z.object({
  v: z.number(),
  type: z.string(),
  project_id: z.string(),
  seq: z.number(),
  ts: z.number(),
  stage: StageSchema,
  event: EventPayloadSchema,
});

export type EventMessage = z.infer<typeof EventMessageSchema>;

// ============================================================================
// EVENT PAYLOAD SCHEMAS (Typed Payloads)
// ============================================================================

// --- Global Event Payloads ---

export const StageStatusPayloadSchema = z.object({
  stage_id: StageIDSchema,
  status: StageStatusSchema,
  message: z.string(),
});

export type StageStatusPayload = z.infer<typeof StageStatusPayloadSchema>;

export const WaitingConfirmationPayloadSchema = z.object({
  stage_id: StageIDSchema,
  summary: z.string(),
  next_actions: z.array(z.string()),
});

export type WaitingConfirmationPayload = z.infer<typeof WaitingConfirmationPayloadSchema>;

export const PlanProposedPayloadSchema = z.object({
  stage_id: StageIDSchema,
  plan_json: z.record(z.string(), z.any()),
});

export type PlanProposedPayload = z.infer<typeof PlanProposedPayloadSchema>;

export const PlanApprovedPayloadSchema = z.object({
  stage_id: StageIDSchema,
});

export type PlanApprovedPayload = z.infer<typeof PlanApprovedPayloadSchema>;

export const FileInfoSchema = z.object({
  path: z.string(),
  type: z.string(),
  size: z.number(),
  sha: z.string().optional(),
});

export type FileInfo = z.infer<typeof FileInfoSchema>;

export const FileTreeUpdatePayloadSchema = z.object({
  files: z.array(FileInfoSchema),
});

export type FileTreeUpdatePayload = z.infer<typeof FileTreeUpdatePayloadSchema>;

export const ArtifactInfoSchema = z.object({
  id: z.string(),
  type: z.string(),
  name: z.string(),
  url: z.string(),
  meta: z.record(z.string(), z.any()).optional(),
});

export type ArtifactInfo = z.infer<typeof ArtifactInfoSchema>;

export const ArtifactAddedPayloadSchema = z.object({
  artifact: ArtifactInfoSchema,
});

export type ArtifactAddedPayload = z.infer<typeof ArtifactAddedPayloadSchema>;

// --- Stage 1: DATA COLLECTION / MODEL CHOICE ---

export const PromptParsedPayloadSchema = z.object({
  task_type: z.string(),
  target: z.string().optional(),
  dataset_hint: z.string().optional(),
  constraints: z.record(z.string(), z.any()).optional(),
});

export type PromptParsedPayload = z.infer<typeof PromptParsedPayloadSchema>;

export const DatasetInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  source: z.string(),
  desc: z.string(),
  meta: z.record(z.string(), z.any()).optional(),
});

export type DatasetInfo = z.infer<typeof DatasetInfoSchema>;

export const DatasetCandidatesPayloadSchema = z.object({
  datasets: z.array(DatasetInfoSchema),
});

export type DatasetCandidatesPayload = z.infer<typeof DatasetCandidatesPayloadSchema>;

export const DatasetSelectedPayloadSchema = z.object({
  dataset_id: z.string(),
});

export type DatasetSelectedPayload = z.infer<typeof DatasetSelectedPayloadSchema>;

export const ModelInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  family: z.string(),
  why: z.string(),
  requirements: z.record(z.string(), z.any()).optional(),
});

export type ModelInfo = z.infer<typeof ModelInfoSchema>;

export const ModelCandidatesPayloadSchema = z.object({
  models: z.array(ModelInfoSchema),
});

export type ModelCandidatesPayload = z.infer<typeof ModelCandidatesPayloadSchema>;

export const ModelSelectedPayloadSchema = z.object({
  model_id: z.string(),
});

export type ModelSelectedPayload = z.infer<typeof ModelSelectedPayloadSchema>;

export const DatasetSampleReadyPayloadSchema = z.object({
  asset_url: z.string(),
  columns: z.array(z.string()),
  n_rows: z.number(),
});

export type DatasetSampleReadyPayload = z.infer<typeof DatasetSampleReadyPayloadSchema>;

// --- Stage 2: PROFILING / PREPROCESSING ---

export const ProfileProgressPayloadSchema = z.object({
  phase: z.string(),
  pct: z.number(),
});

export type ProfileProgressPayload = z.infer<typeof ProfileProgressPayloadSchema>;

export const ProfileSummaryPayloadSchema = z.object({
  n_rows: z.number(),
  n_cols: z.number(),
  missing_pct: z.number(),
  types_breakdown: z.record(z.string(), z.number()),
  warnings: z.array(z.string()),
});

export type ProfileSummaryPayload = z.infer<typeof ProfileSummaryPayloadSchema>;

export const MissingnessTableReadyPayloadSchema = z.object({
  asset_url: z.string(),
});

export type MissingnessTableReadyPayload = z.infer<typeof MissingnessTableReadyPayloadSchema>;

export const TargetDistributionReadyPayloadSchema = z.object({
  asset_url: z.string(),
});

export type TargetDistributionReadyPayload = z.infer<typeof TargetDistributionReadyPayloadSchema>;

export const SplitSummaryPayloadSchema = z.object({
  train_rows: z.number(),
  val_rows: z.number(),
  test_rows: z.number(),
  stratified: z.boolean(),
  seed: z.number(),
});

export type SplitSummaryPayload = z.infer<typeof SplitSummaryPayloadSchema>;

export const PreprocessPlanPayloadSchema = z.object({
  steps: z.array(z.record(z.string(), z.any())),
});

export type PreprocessPlanPayload = z.infer<typeof PreprocessPlanPayloadSchema>;

// --- Stage 3: TRAINING (RICH) ---

export const TrainRunStartedPayloadSchema = z.object({
  run_id: z.string(),
  model_id: z.string(),
  metric_primary: z.string(),
  config: z.record(z.string(), z.any()),
});

export type TrainRunStartedPayload = z.infer<typeof TrainRunStartedPayloadSchema>;

export const TrainProgressPayloadSchema = z.object({
  run_id: z.string(),
  epoch: z.number(),
  epochs: z.number(),
  step: z.number(),
  steps: z.number(),
  eta_s: z.number(),
  phase: z.string(),
});

export type TrainProgressPayload = z.infer<typeof TrainProgressPayloadSchema>;

export const MetricScalarPayloadSchema = z.object({
  run_id: z.string(),
  name: z.string(),
  split: z.string(),
  step: z.number(),
  value: z.number(),
});

export type MetricScalarPayload = z.infer<typeof MetricScalarPayloadSchema>;

export const LeaderboardRowSchema = z.object({
  model: z.string(),
  params: z.record(z.string(), z.any()),
  metric: z.number(),
  runtime_s: z.number(),
});

export type LeaderboardRow = z.infer<typeof LeaderboardRowSchema>;

export const LeaderboardUpdatedPayloadSchema = z.object({
  rows: z.array(LeaderboardRowSchema),
});

export type LeaderboardUpdatedPayload = z.infer<typeof LeaderboardUpdatedPayloadSchema>;

export const BestModelUpdatedPayloadSchema = z.object({
  run_id: z.string(),
  model_id: z.string(),
  metric: z.number(),
});

export type BestModelUpdatedPayload = z.infer<typeof BestModelUpdatedPayloadSchema>;

export const ConfusionMatrixReadyPayloadSchema = z.object({
  asset_url: z.string(),
});

export type ConfusionMatrixReadyPayload = z.infer<typeof ConfusionMatrixReadyPayloadSchema>;

export const ROCCurveReadyPayloadSchema = z.object({
  asset_url: z.string(),
});

export type ROCCurveReadyPayload = z.infer<typeof ROCCurveReadyPayloadSchema>;

export const ResidualsPlotReadyPayloadSchema = z.object({
  asset_url: z.string(),
});

export type ResidualsPlotReadyPayload = z.infer<typeof ResidualsPlotReadyPayloadSchema>;

export const FeatureImportanceReadyPayloadSchema = z.object({
  asset_url: z.string(),
});

export type FeatureImportanceReadyPayload = z.infer<typeof FeatureImportanceReadyPayloadSchema>;

export const ResourceStatsPayloadSchema = z.object({
  run_id: z.string(),
  cpu_pct: z.number(),
  ram_mb: z.number(),
  gpu_pct: z.number().optional(),
  vram_mb: z.number().optional(),
  step_per_sec: z.number().optional(),
});

export type ResourceStatsPayload = z.infer<typeof ResourceStatsPayloadSchema>;

export const LogLinePayloadSchema = z.object({
  run_id: z.string(),
  level: z.string(),
  text: z.string(),
});

export type LogLinePayload = z.infer<typeof LogLinePayloadSchema>;

export const TrainRunFinishedPayloadSchema = z.object({
  run_id: z.string(),
  status: z.string(),
  final_metrics: z.record(z.string(), z.number()),
});

export type TrainRunFinishedPayload = z.infer<typeof TrainRunFinishedPayloadSchema>;

// --- Stage 4: REVIEW / EDIT ---

export const ReportReadyPayloadSchema = z.object({
  asset_url: z.string(),
});

export type ReportReadyPayload = z.infer<typeof ReportReadyPayloadSchema>;

export const NotebookReadyPayloadSchema = z.object({
  asset_url: z.string(),
});

export type NotebookReadyPayload = z.infer<typeof NotebookReadyPayloadSchema>;

export const CodeWorkspaceReadyPayloadSchema = z.object({
  files: z.array(FileInfoSchema),
});

export type CodeWorkspaceReadyPayload = z.infer<typeof CodeWorkspaceReadyPayloadSchema>;

export const EditSuggestionsPayloadSchema = z.object({
  suggestions: z.array(z.record(z.string(), z.any())),
});

export type EditSuggestionsPayload = z.infer<typeof EditSuggestionsPayloadSchema>;

// --- Stage 5: EXPORT ---

export const ExportProgressPayloadSchema = z.object({
  pct: z.number(),
  message: z.string(),
});

export type ExportProgressPayload = z.infer<typeof ExportProgressPayloadSchema>;

export const ExportReadyPayloadSchema = z.object({
  asset_url: z.string(),
  contents: z.array(z.string()),
  checksum: z.string(),
});

export type ExportReadyPayload = z.infer<typeof ExportReadyPayloadSchema>;

// ============================================================================
// STATE SNAPSHOT
// ============================================================================

export const StateSnapshotSchema = z.object({
  project_id: z.string(),
  stage: StageSchema,
  decisions: z.record(z.string(), z.any()),
  plans: z.record(z.string(), z.any()),
  artifacts: z.array(ArtifactInfoSchema),
  files: z.array(FileInfoSchema),
  ui_layout: z.record(z.string(), z.any()),
});

export type StateSnapshot = z.infer<typeof StateSnapshotSchema>;

// ============================================================================
// UTILITY TYPE HELPERS
// ============================================================================

/**
 * Type-safe event payload extractor with runtime validation
 * Usage: const payload = extractPayload(eventMessage, DatasetSampleReadyPayloadSchema);
 */
export function extractPayload<T extends z.ZodTypeAny>(
  eventMessage: EventMessage,
  schema: T
): z.infer<T> {
  return schema.parse(eventMessage.event.payload);
}

/**
 * Type guard to check event type
 * Usage: if (isEventType(msg, "DATASET_SAMPLE_READY")) { ... }
 */
export function isEventType(
  eventMessage: EventMessage,
  eventType: EventType
): boolean {
  return eventMessage.event.name === eventType;
}

/**
 * Validate and parse incoming WebSocket message
 * Usage: const message = parseEventMessage(rawData);
 */
export function parseEventMessage(data: unknown): EventMessage {
  return EventMessageSchema.parse(data);
}
