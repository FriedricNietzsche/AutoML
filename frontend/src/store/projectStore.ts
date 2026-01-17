import { create } from 'zustand';

import { joinUrl, resolveHttpBase } from '../lib/api';
import {
  STAGE_ORDER,
  type EventType,
  type StageID,
  type StageStatus,
  type StageStatusPayload,
  type WaitingConfirmationPayload,
} from '../lib/contract';
import { createWebSocketClient, type ConnectionStatus, type EventEnvelope, type WSClient } from '../lib/ws';

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
  if (!incoming || incoming.length === 0) return prev;
  const next = { ...prev };
  for (const stage of incoming) {
    const normalized = normalizeStageState(stage);
    if (!normalized) continue;
    next[normalized.id] = { ...next[normalized.id], ...normalized };
  }
  return next;
};

const MAX_EVENTS = 300;

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
  connect: (opts?: { projectId?: string; wsBase?: string }) => void;
  disconnect: () => void;
  hydrate: () => Promise<void>;
  confirm: () => Promise<void>;
  applyEvent: (evt: EventEnvelope) => void;
};

export const useProjectStore = create<ProjectStoreState>((set, get) => ({
  projectId: 'demo-project',
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

  connect: (opts) => {
    const projectId = opts?.projectId ?? get().projectId;
    const wsBase = opts?.wsBase ?? get().wsBase;
    const apiBase = resolveHttpBase(wsBase);

    get().wsClient?.close();

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
      const incomingStages = mergeStages(get().stages, json.stages);
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
      set({
        stages: mergeStages(get().stages, json.stages),
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
    const payload = evt.event?.payload as StageStatusPayload | WaitingConfirmationPayload | undefined;

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

    if (name === 'WAITING_CONFIRMATION' && payload) {
      const waitPayload = payload as WaitingConfirmationPayload;
      set({ waitingConfirmation: waitPayload });
    }

    set((state) => ({
      lastEvent: evt,
      events: [...state.events, evt].slice(-MAX_EVENTS),
    }));
  },
}));
