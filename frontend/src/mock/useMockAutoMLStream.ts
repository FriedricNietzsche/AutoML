import { useEffect, useMemo, useRef, useState } from 'react';
import type { MockWSEnvelope } from './backendEventTypes';
import { createMockAutoMLStream } from './mockBackendStream';
import type { ScenarioId } from './scenarios';
import { 
  reduceMetricsState, 
  emptyMetrics, 
  type MetricsState 
} from '../lib/metricsReducer';

export type { MetricsState };

export function useMockAutoMLStream({ scenarioId, seed, enabled }: { scenarioId: ScenarioId; seed?: number; enabled?: boolean }) {
  const [events, setEvents] = useState<MockWSEnvelope[]>([]);
  const [metricsState, setMetricsState] = useState<MetricsState>(emptyMetrics);
  const latestByType = useMemo(() => {
    const map = new Map<string, MockWSEnvelope>();
    for (const event of events) {
      const key = event.event?.name ?? event.type ?? 'EVENT';
      map.set(key, event);
    }
    return map;
  }, [events]);

  const runIdRef = useRef(0);
  const pendingEventsRef = useRef<MockWSEnvelope[]>([]);
  const pendingMetricsRef = useRef<MetricsState>(emptyMetrics);
  const assetMapRef = useRef(new Map<string, Record<string, unknown>>());
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled) return;
    runIdRef.current += 1;
    const runId = runIdRef.current;
    setEvents([]);
    setMetricsState(emptyMetrics);
    pendingEventsRef.current = [];
    pendingMetricsRef.current = emptyMetrics;
    assetMapRef.current = new Map();
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const stream = createMockAutoMLStream({ scenarioId, seed });

    const flush = () => {
      rafRef.current = null;
      const batch = pendingEventsRef.current;
      if (batch.length === 0) return;
      pendingEventsRef.current = [];
      const nextMetrics = pendingMetricsRef.current;
      setEvents((prev) => [...prev, ...batch]);
      setMetricsState(nextMetrics);
    };

    (async () => {
      for await (const event of stream) {
        if (runIdRef.current !== runId) return;
        pendingEventsRef.current.push(event);
        pendingMetricsRef.current = reduceMetricsState(pendingMetricsRef.current, event, assetMapRef.current);
        if (rafRef.current === null) {
          rafRef.current = requestAnimationFrame(flush);
        }
      }
    })();

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [enabled, scenarioId, seed]);

  return { events, latestByType, metricsState };
}
