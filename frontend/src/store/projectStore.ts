import { create } from 'zustand';

import { joinUrl, resolveHttpBase } from '../lib/api';
import {
  STAGE_ORDER,
  type EventType,
  type StageID,
  type StageStatus,
  type StageStatusPayload,
  type WaitingConfirmationPayload,
  type DatasetSampleReadyPayload,
  type TrainProgressPayload,
  type MetricScalarPayload,
  type ArtifactAddedPayload,
  type ProfileSummaryPayload,
} from '../lib/contract';
import { createWebSocketClient, type ConnectionStatus, type EventEnvelope, type WSClient } from '../lib/ws';

const DEFAULT_PROJECT_ID =
  (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_PROJECT_ID) || 'demo';

type StageState = {
  id: StageID;
  index: number;
  status: StageStatus;
  message?: string;
};

type ProjectSnapshot = {
  project_id: string;
  current_stage?: StageState;
  stages?: StageState[];
  waiting_confirmation?: WaitingConfirmationPayload | null;
};

const STAGE_IDS = STAGE_ORDER.map((s) => s.id);
const isStageId = (value: unknown): value is StageID => typeof value === 'string' && STAGE_IDS.includes(value as StageID);
const STAGE_STATUS_VALUES: StageStatus[] = ['PENDING', 'IN_PROGRESS', 'WAITING_CONFIRMATION', 'COMPLETED', 'FAILED', 'SKIPPED'];
const isStageStatus = (value: unknown): value is StageStatus =>
  typeof value === 'string' && STAGE_STATUS_VALUES.includes(value as StageStatus);

const buildDefaultStages = (): Record<StageID, StageState> =>
  STAGE_ORDER.reduce((acc, stageDef, idx) => {
    acc[stageDef.id] = { id: stageDef.id, index: idx, status: idx === 0 ? 'IN_PROGRESS' : 'PENDING' };
    return acc;
  }, {} as Record<StageID, StageState>);

const normalizeStageState = (stage: any): StageState | null => {
  if (!stage) return null;
  if (!isStageId(stage.id)) return null;
  const status = isStageStatus(stage.status) ? stage.status : 'PENDING';
  const index =
    typeof stage.index === 'number'
      ? stage.index
      : Math.max(
        STAGE_ORDER.findIndex((s) => s.id === stage.id),
        0
      );
  return { id: stage.id, status, index, message: typeof stage.message === 'string' ? stage.message : undefined };
};

const mergeStages = (prev: Record<StageID, StageState>, incoming?: StageState[]) => {
  if (!incoming || !Array.isArray(incoming) || incoming.length === 0) { console.debug("[projectStore] incoming not array:", typeof incoming); return prev; }
  const next = { ...prev };
  for (const stage of incoming) {
    const normalized = normalizeStageState(stage);
    if (!normalized) continue;
    next[normalized.id] = { ...next[normalized.id], ...normalized };
  }
  return next;
};

const MAX_EVENTS = 300;

// Extended state for data-specific events
export type DatasetSample = {
  assetUrl: string;
  columns: string[];
  nRows: number;
  images?: string[];  // For image datasets
  sample_rows?: any[];  // For tabular datasets - actual row data
};

export type TrainingMetrics = {
  runId: string;
  metricsHistory: Array<{ step: number; name: string; split: string; value: number }>;
  progress?: TrainProgressPayload;
  artifacts: Array<{ id: string; type: string; name: string; url: string }>;
};

export type GDPathState = {
  runId: string;
  surfaceSpec?: any;
  domainHalf?: number;
  points: Array<{ x: number; y: number }>;
  finished: boolean;
};

type ProjectStoreState = {
  projectId: string;
  wsBase?: string;
  apiBase: string;
  connectionStatus: ConnectionStatus;
  stages: Record<StageID, StageState>;
  currentStageId: StageID;
  waitingConfirmation: WaitingConfirmationPayload | null;
  events: EventEnvelope[];
  lastEvent: EventEnvelope | null;
  error: string | null;
  wsClient: WSClient | null;

  // Extended state
  datasetSample: DatasetSample | null;
  profileSummary: ProfileSummaryPayload | null;
  trainingMetrics: TrainingMetrics | null;
  gdPath: GDPathState | null;
  artifacts: Array<{ id: string; type: string; name: string; url: string; meta?: any }>;

  connect: (opts?: { projectId?: string; wsBase?: string }) => void;
  disconnect: () => void;
  hydrate: () => Promise<void>;
  confirm: () => Promise<void>;
  applyEvent: (evt: EventEnvelope) => void;
};

export const useProjectStore = create<ProjectStoreState>((set, get) => ({
  projectId: DEFAULT_PROJECT_ID,
  wsBase: undefined,
  apiBase: resolveHttpBase(),
  connectionStatus: 'idle',
  stages: buildDefaultStages(),
  currentStageId: STAGE_ORDER[0].id,
  waitingConfirmation: null,
  events: [],
  lastEvent: null,
  error: null,
  wsClient: null,
  datasetSample: null,
  profileSummary: null,
  trainingMetrics: null,
  gdPath: null,
  artifacts: [],

  connect: (opts) => {
    const { projectId: currentId, wsBase: currentBase, wsClient } = get();
    const projectId = opts?.projectId ?? currentId;
    const wsBase = opts?.wsBase ?? currentBase;
    const apiBase = resolveHttpBase(wsBase);

    // Skip if already connected to the same project
    if (wsClient && currentId === projectId && currentBase === wsBase && (get().connectionStatus === 'open' || get().connectionStatus === 'connecting')) {
      return;
    }

    wsClient?.close();


    const client = createWebSocketClient({
      projectId,
      baseUrl: wsBase,
      onStatusChange: (status) => set({ connectionStatus: status }),
      onEvent: (evt) => get().applyEvent(evt),
      onError: (err) =>
        set({
          error: err instanceof Error ? err.message : String(err),
          connectionStatus: 'error',
        }),
    });

    set({
      projectId,
      wsBase,
      apiBase,
      wsClient: client,
      connectionStatus: 'connecting',
      events: [],
      lastEvent: null,
      error: null,
      waitingConfirmation: null,
    });
  },

  disconnect: () => {
    get().wsClient?.close();
    set({ wsClient: null, connectionStatus: 'closed' });
  },

  hydrate: async () => {
    const { apiBase, projectId } = get();
    try {
      const res = await fetch(joinUrl(apiBase, `/api/projects/${projectId}/state`));
      if (!res.ok) throw new Error(`State fetch failed (${res.status})`);
      const json = (await res.json()) as ProjectSnapshot;
      // Convert stages from object to array if needed
      const stagesArray = json.stages && typeof json.stages === 'object' && !Array.isArray(json.stages)
        ? Object.values(json.stages)
        : json.stages;
      const incomingStages = mergeStages(get().stages, stagesArray as any);
      const nextCurrent = json.current_stage && isStageId(json.current_stage.id) ? json.current_stage.id : get().currentStageId;
      set({
        stages: incomingStages,
        currentStageId: nextCurrent,
        waitingConfirmation: json.waiting_confirmation ?? null,
        error: null,
      });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : String(err) });
    }
  },

  confirm: async () => {
    const { apiBase, projectId } = get();
    try {
      const res = await fetch(joinUrl(apiBase, `/api/projects/${projectId}/confirm`), { method: 'POST' });
      if (!res.ok) throw new Error(`Confirm failed (${res.status})`);
      const json = (await res.json()) as ProjectSnapshot;
      // Convert stages from object to array if needed
      const stagesArray2 = json.stages && typeof json.stages === 'object' && !Array.isArray(json.stages)
        ? Object.values(json.stages)
        : json.stages;
      set({
        stages: mergeStages(get().stages, stagesArray2 as any),
        currentStageId:
          json.current_stage && isStageId(json.current_stage.id) ? json.current_stage.id : get().currentStageId,
        waitingConfirmation: json.waiting_confirmation ?? null,
        error: null,
      });
    } catch (err) {
      set({ error: err instanceof Error ? err.message : String(err) });
    }
  },

  applyEvent: (evt: EventEnvelope) => {
    const name = evt.event?.name as EventType | undefined;
    const payload = evt.event?.payload as any;

    // Handle STAGE_STATUS
    if (name === 'STAGE_STATUS' && payload && isStageId((payload as StageStatusPayload).stage_id)) {
      const stagePayload = payload as StageStatusPayload;
      if (isStageStatus(stagePayload.status)) {
        set((state) => {
          const existing = state.stages[stagePayload.stage_id] ?? {
            id: stagePayload.stage_id,
            index: STAGE_ORDER.findIndex((s) => s.id === stagePayload.stage_id),
            status: 'PENDING' as StageStatus,
          };
          const updated: StageState = {
            ...existing,
            status: stagePayload.status,
            message: stagePayload.message,
          };
          return {
            stages: { ...state.stages, [stagePayload.stage_id]: updated },
            currentStageId: stagePayload.stage_id,
            waitingConfirmation:
              stagePayload.status === 'WAITING_CONFIRMATION' ? state.waitingConfirmation : null,
          };
        });
      }
    }

    // Handle WAITING_CONFIRMATION
    if (name === 'WAITING_CONFIRMATION' && payload) {
      const waitPayload = payload as WaitingConfirmationPayload;
      set({ waitingConfirmation: waitPayload });
    }

    // Handle DATASET_SAMPLE_READY
    if (name === 'DATASET_SAMPLE_READY' && payload) {
      const sample = payload as DatasetSampleReadyPayload;
      set({
        datasetSample: {
          assetUrl: sample.asset_url,
          columns: sample.columns || [],
          nRows: sample.n_rows || 0,
          images: (payload as any).images || [],
          sample_rows: (payload as any).sample_rows || [],  // Capture sample rows
        },
      });
    }

    // Handle PROFILE_SUMMARY
    if (name === 'PROFILE_SUMMARY' && payload) {
      set({ profileSummary: payload as ProfileSummaryPayload });
    }

    // Handle TRAIN_RUN_STARTED
    if (name === 'TRAIN_RUN_STARTED' && payload) {
      set({
        trainingMetrics: {
          runId: payload.run_id,
          metricsHistory: [],
          artifacts: [],
        },
      });
    }

    // Handle TRAIN_PROGRESS
    if (name === 'TRAIN_PROGRESS' && payload) {
      set((state) => ({
        trainingMetrics: state.trainingMetrics
          ? { ...state.trainingMetrics, progress: payload as TrainProgressPayload }
          : null,
      }));
    }

    // Handle METRIC_SCALAR
    if (name === 'METRIC_SCALAR' && payload) {
      const metric = payload as MetricScalarPayload;
      set((state) => ({
        trainingMetrics: state.trainingMetrics
          ? {
            ...state.trainingMetrics,
            metricsHistory: [...state.trainingMetrics.metricsHistory, {
              step: metric.step,
              name: metric.name,
              split: metric.split,
              value: metric.value,
            }],
          }
          : null,
      }));
    }

    // Handle ARTIFACT_ADDED
    if (name === 'ARTIFACT_ADDED' && payload) {
      const artifact = (payload as ArtifactAddedPayload).artifact;
      set((state) => ({
        artifacts: [...state.artifacts, artifact],
        trainingMetrics: state.trainingMetrics
          ? {
            ...state.trainingMetrics,
            artifacts: [...state.trainingMetrics.artifacts, artifact],
          }
          : state.trainingMetrics,
      }));
    }

    // Handle GD visualization events
    if (name === 'LOSS_SURFACE_SPEC_READY' && payload) {
      set({
        gdPath: {
          runId: payload.run_id,
          surfaceSpec: payload.surface_spec,
          domainHalf: payload.surface_spec?.domainHalf,
          points: [],
          finished: false,
        },
      });
    }

    if (name === 'GD_PATH_STARTED' && payload) {
      set((state) => ({
        gdPath: state.gdPath
          ? { ...state.gdPath, domainHalf: payload.domainHalf, points: payload.point0 ? [payload.point0] : [] }
          : null,
      }));
    }

    if (name === 'GD_PATH_UPDATE' && payload) {
      set((state) => ({
        gdPath: state.gdPath
          ? { ...state.gdPath, points: [...state.gdPath.points, ...(payload.points || [])] }
          : null,
      }));
    }

    if (name === 'GD_PATH_FINISHED' && payload) {
      set((state) => ({
        gdPath: state.gdPath ? { ...state.gdPath, finished: true } : null,
      }));
    }

    set((state) => ({
      lastEvent: evt,
      events: [...state.events, evt].slice(-MAX_EVENTS),
    }));
  },
}));
