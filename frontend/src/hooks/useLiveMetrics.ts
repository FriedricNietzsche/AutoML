import { useEffect, useState, useRef } from 'react';
import { useProjectStore } from '../store/projectStore';
import { 
  reduceMetricsState, 
  emptyMetrics, 
  type MetricsState 
} from '../lib/metricsReducer';

/**
 * Hook that listens to the global project store and reduces
 * WebSocket events into a metrics state for visualization.
 */
export function useLiveMetrics() {
  const events = useProjectStore((state) => state.events);
  const [metricsState, setMetricsState] = useState<MetricsState>(emptyMetrics);
  
  const processedCountRef = useRef(0);
  const assetMapRef = useRef(new Map<string, Record<string, unknown>>());
  const pendingMetricsRef = useRef<MetricsState>(emptyMetrics);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    // If events were reset (e.g. on new connection)
    if (events.length === 0) {
      setMetricsState(emptyMetrics);
      pendingMetricsRef.current = emptyMetrics;
      processedCountRef.current = 0;
      assetMapRef.current = new Map();
      return;
    }

    // Process new events
    if (events.length > processedCountRef.current) {
      let currentMetrics = pendingMetricsRef.current;
      
      for (let i = processedCountRef.current; i < events.length; i++) {
        currentMetrics = reduceMetricsState(currentMetrics, events[i], assetMapRef.current);
      }
      
      pendingMetricsRef.current = currentMetrics;
      processedCountRef.current = events.length;

      // Batch state updates using requestAnimationFrame
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(() => {
          setMetricsState(pendingMetricsRef.current);
          rafRef.current = null;
        });
      }
    }

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [events]);

  return { metricsState };
}
