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

export type EventType = 'HELLO' | 'STAGE_STATUS' | 'WAITING_CONFIRMATION' | 'LOG_LINE' | 'PLAN_PROPOSED' | 'PLAN_APPROVED';

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
