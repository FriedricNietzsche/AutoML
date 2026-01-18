import { useEffect, useRef } from 'react';
import type { MockWSEnvelope } from '../../../../mock/backendEventTypes';
import type { ArtifactAddedPayload, LogLinePayload } from '../../../../lib/contract';
import { appendLog } from '../utils/loaderHelpers';

interface UseEventProcessorProps {
  events: MockWSEnvelope[];
  useMockStream: boolean;
  updateFileContent: (path: string, content: string | ((prev: string) => string)) => void;
  applyProjectEvent: (event: MockWSEnvelope) => void;
}

/**
 * Custom hook that processes incoming WebSocket events
 * Handles artifact creation and log line events.
 * 
 * Works for both mock and live streams.
 */
export function useEventProcessor({
  events,
  useMockStream, // Still passed but guard removed for side-effects
  updateFileContent,
  applyProjectEvent,
}: UseEventProcessorProps) {
  const processedEventsRef = useRef(0);
  const updateFileContentRef = useRef(updateFileContent);

  useEffect(() => {
    updateFileContentRef.current = updateFileContent;
  }, [updateFileContent]);

  useEffect(() => {
    // We process events regardless of useMockStream to support real backend side-effects
    if (events.length <= processedEventsRef.current) return;

    for (let i = processedEventsRef.current; i < events.length; i += 1) {
      const event = events[i] as MockWSEnvelope;
      
      // In mock mode, we manually apply the event to the project store
      // In live mode, the store already has it, so this might be redundant 
      // but safe if applyProjectEvent handles duplicates.
      if (useMockStream) {
        applyProjectEvent(event);
      }
      
      const name = event.event?.name;
      const payload = event.event?.payload as Record<string, unknown> | undefined;

      if (name === 'ARTIFACT_ADDED') {
        const artifactPayload = payload as ArtifactAddedPayload | undefined;
        const artifact = artifactPayload?.artifact;
        const meta = artifact?.meta as Record<string, unknown> | undefined;
        const filePath = meta?.file_path as string | undefined;
        const content = meta?.content as string | undefined;
        if (filePath && typeof content === 'string') {
          updateFileContentRef.current(filePath, content);
        }
      }

      if (name === 'LOG_LINE') {
        const logPayload = payload as LogLinePayload | undefined;
        if (!logPayload?.text) continue;
        appendLog(updateFileContentRef.current, logPayload.text, logPayload.level);
      }
    }

    processedEventsRef.current = events.length;
  }, [events, useMockStream, applyProjectEvent]);
}
