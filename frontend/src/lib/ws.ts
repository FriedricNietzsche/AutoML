import type { WSEnvelope } from './contract';

export type ConnectionStatus = 'idle' | 'connecting' | 'open' | 'closed' | 'error';
export type EventEnvelope = WSEnvelope;

type WSClientOptions = {
  projectId: string;
  baseUrl?: string; // e.g. ws://localhost:8000
  retryDelaysMs?: number[];
  onStatusChange?: (status: ConnectionStatus) => void;
  onEvent?: (event: EventEnvelope) => void;
  onError?: (err: unknown) => void;
};

export type WSClient = {
  send: (data: unknown) => void;
  close: () => void;
};

const defaultRetry = [500, 1000, 2000, 5000];

const resolveBaseUrl = () => {
  // Allow override via env (set VITE_WS_BASE or VITE_BACKEND_WS_BASE)
  const envBase =
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_WS_BASE) ||
    (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_BACKEND_WS_BASE);
  if (envBase) return envBase as string;

  // Dev fallback: assume backend on 8000 when frontend runs on 5173.
  if (typeof window !== 'undefined') {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    return `${proto}://${window.location.hostname}:8000`;
  }
  return 'ws://localhost:8000';
};

const makeUrl = (baseUrl: string, projectId: string) => {
  const trimmed = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  return `${trimmed}/ws/projects/${projectId}`;
};

export function createWebSocketClient(opts: WSClientOptions): WSClient {
  const {
    projectId,
    baseUrl,
    retryDelaysMs = defaultRetry,
    onStatusChange,
    onEvent,
    onError,
  } = opts;

  let socket: WebSocket | null = null;
  let closed = false;
  let attempt = 0;

  const url = baseUrl ?? resolveBaseUrl();

  const fullUrl = makeUrl(url, projectId);

  const connect = () => {
    onStatusChange?.('connecting');
    socket = new WebSocket(fullUrl);

    socket.onopen = () => {
      attempt = 0;
      onStatusChange?.('open');
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(event.data) as EventEnvelope;
        onEvent?.(parsed);
      } catch (err) {
        onError?.(err);
      }
    };

    socket.onerror = (err) => {
      onStatusChange?.('error');
      onError?.(err);
    };

    socket.onclose = () => {
      onStatusChange?.('closed');
      if (!closed) {
        const delay = retryDelaysMs[Math.min(attempt, retryDelaysMs.length - 1)];
        attempt += 1;
        setTimeout(connect, delay);
      }
    };
  };

  connect();

  const send = (data: unknown) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(data));
    }
  };

  const close = () => {
    closed = true;
    if (socket && socket.readyState !== WebSocket.CLOSED) {
      socket.close();
    }
  };

  return { send, close };
}

export function isHello(event: EventEnvelope) {
  return event.event?.name === 'HELLO';
}
