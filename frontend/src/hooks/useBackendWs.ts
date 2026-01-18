import { useEffect, useMemo, useState } from 'react';
import type { ConnectionStatus, EventEnvelope } from '../lib/ws';
import { createWebSocketClient } from '../lib/ws';

const DEFAULT_PROJECT_ID =
  (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_PROJECT_ID) || 'demo';

type UseBackendWsOptions = {
  projectId?: string;
  baseUrl?: string;
};

const formatLine = (evt: EventEnvelope) => {
  const ts = evt.ts ?? Date.now();
  const name = evt.event?.name ?? evt.type ?? 'event';
  const payload = evt.event?.payload ? JSON.stringify(evt.event.payload) : '';
  const stage = evt.stage?.id ? ` [${evt.stage.id}]` : '';
  return `${new Date(ts).toISOString()} [${name}]${stage} ${payload}`.trim();
};

export function useBackendWs(options?: UseBackendWsOptions) {
  const { projectId = DEFAULT_PROJECT_ID, baseUrl } = options ?? {};

  const [status, setStatus] = useState<ConnectionStatus>('idle');
  const [events, setEvents] = useState<EventEnvelope[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setEvents([]);
    setError(null);
    const client = createWebSocketClient({
      projectId,
      baseUrl,
      onStatusChange: setStatus,
      onEvent: (evt) => {
        setEvents((prev) => [...prev, evt].slice(-300));
      },
      onError: (err) => {
        setError(err instanceof Error ? err.message : String(err));
      },
    });
    return () => client.close();
  }, [projectId, baseUrl]);

  const lastEvent = events.length > 0 ? events[events.length - 1] : null;
  const logText = useMemo(() => events.map(formatLine).join('\n'), [events]);

  return { status, events, lastEvent, logText, error };
}
