import { useEffect, useReducer, useCallback, useRef } from 'react';

const WS_BASE = import.meta.env.VITE_WS_BASE || 'ws://localhost:8000';

interface WSEvent {
  type: string;
  [key: string]: unknown;
}

interface MetricsState {
  events: WSEvent[];
  metrics: Record<string, number>[];
  gradients: { epoch: number; layer: string; mean: number; std: number }[];
  embeddings: { x: number; y: number; label: string }[];
  confusionMatrix: number[][] | null;
  featureImportances: { feature: string; importance: number }[] | null;
  previewImages: string[];
  isConnected: boolean;
  connectionError: string | null;
}

type Action =
  | { type: 'WS_OPEN' }
  | { type: 'WS_CLOSE' }
  | { type: 'WS_ERROR'; error: string }
  | { type: 'WS_MESSAGE'; event: WSEvent }
  | { type: 'RESET' };

const initialState: MetricsState = {
  events: [],
  metrics: [],
  gradients: [],
  embeddings: [],
  confusionMatrix: null,
  featureImportances: null,
  previewImages: [],
  isConnected: false,
  connectionError: null,
};

function reducer(state: MetricsState, action: Action): MetricsState {
  switch (action.type) {
    case 'WS_OPEN':
      return { ...state, isConnected: true, connectionError: null };
    
    case 'WS_CLOSE':
      return { ...state, isConnected: false };
    
    case 'WS_ERROR':
      return { ...state, isConnected: false, connectionError: action.error };
    
    case 'WS_MESSAGE': {
      const event = action.event;
      const newState = {
        ...state,
        events: [...state.events, event].slice(-100), // Keep last 100 events
      };

      // Route event to appropriate state slice
      switch (event.type) {
        case 'METRIC':
          if (event.data && typeof event.data === 'object') {
            newState.metrics = [...state.metrics, event.data as Record<string, number>].slice(-50);
          }
          break;

        case 'GRADIENT':
          if (event.epoch !== undefined) {
            newState.gradients = [
              ...state.gradients,
              {
                epoch: event.epoch as number,
                layer: (event.layer as string) || 'unknown',
                mean: (event.mean as number) || 0,
                std: (event.std as number) || 0,
              },
            ].slice(-200);
          }
          break;

        case 'EMBEDDING':
          if (Array.isArray(event.points)) {
            newState.embeddings = event.points as MetricsState['embeddings'];
          }
          break;

        case 'CONFUSION_MATRIX':
          if (Array.isArray(event.matrix)) {
            newState.confusionMatrix = event.matrix as number[][];
          }
          break;

        case 'FEATURE_IMPORTANCE':
          if (Array.isArray(event.features)) {
            newState.featureImportances = event.features as MetricsState['featureImportances'];
          }
          break;

        case 'PREVIEW_IMAGE':
          if (event.url && typeof event.url === 'string') {
            newState.previewImages = [...state.previewImages, event.url].slice(-20);
          }
          break;

        case 'STAGE_STATUS':
          // Already captured in events array
          break;

        default:
          // Unknown event type, just log
          console.debug('Unknown WS event type:', event.type);
      }

      return newState;
    }

    case 'RESET':
      return initialState;

    default:
      return state;
  }
}

export function useLiveMetrics(projectId: string) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (!projectId) return;

    const wsUrl = `${WS_BASE}/ws/${projectId}`;
    console.log('[WS] Connecting to:', wsUrl);

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WS] Connected');
        dispatch({ type: 'WS_OPEN' });
      };

      ws.onmessage = (evt) => {
        try {
          const event = JSON.parse(evt.data) as WSEvent;
          dispatch({ type: 'WS_MESSAGE', event });
        } catch (err) {
          console.error('[WS] Failed to parse message:', err);
        }
      };

      ws.onerror = (evt) => {
        console.error('[WS] Error:', evt);
        dispatch({ type: 'WS_ERROR', error: 'Connection error' });
      };

      ws.onclose = (evt) => {
        console.log('[WS] Closed:', evt.code, evt.reason);
        dispatch({ type: 'WS_CLOSE' });
        
        // Attempt reconnect after 3 seconds
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        reconnectTimeoutRef.current = window.setTimeout(() => {
          console.log('[WS] Attempting reconnect...');
          connect();
        }, 3000);
      };
    } catch (err) {
      console.error('[WS] Failed to create WebSocket:', err);
      dispatch({ type: 'WS_ERROR', error: 'Failed to connect' });
    }
  }, [projectId]);

  useEffect(() => {
    dispatch({ type: 'RESET' });
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return state;
}
