/**
 * Real backend pipeline integration hook
 * Uses the EXISTING WebSocket connection from projectStore
 * NO duplicate connections - reuses the global connection managed by AppShell
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { useProjectStore } from '../store/projectStore';
import type { EventEnvelope } from '../lib/ws';
import type { StageID } from '../lib/contract';

interface PipelineState {
  isRunning: boolean;
  currentStage: StageID | null;
  error: string | null;
  events: EventEnvelope[];
}

interface UsePipelineOptions {
  projectId: string;
  onStageChange?: (stage: StageID) => void;
  onError?: (error: string) => void;
  onComplete?: () => void;
}

export function useBackendPipeline(options: UsePipelineOptions) {
  const { projectId } = options;
  
  // Use refs for callbacks to prevent dependency changes
  const onStageChangeRef = useRef(options.onStageChange);
  const onErrorRef = useRef(options.onError);
  const onCompleteRef = useRef(options.onComplete);
  
  // Update refs when callbacks change
  useEffect(() => {
    onStageChangeRef.current = options.onStageChange;
    onErrorRef.current = options.onError;
    onCompleteRef.current = options.onComplete;
  });
  
  const [state, setState] = useState<PipelineState>({
    isRunning: false,
    currentStage: null,
    error: null,
    events: [],
  });

  // Use the EXISTING WebSocket connection from projectStore (managed by AppShell)
  // DO NOT create a new connection here!
  const { 
    connectionStatus, 
    events: wsEvents, 
    currentStageId,
    stages,
    apiBase,
  } = useProjectStore();

  // Track stage changes
  useEffect(() => {
    if (currentStageId && currentStageId !== state.currentStage) {
      setState(prev => ({ ...prev, currentStage: currentStageId }));
      onStageChangeRef.current?.(currentStageId);
      
      // Check if we've reached EXPORT stage (pipeline complete)
      if (currentStageId === 'EXPORT' && stages[currentStageId]?.status === 'COMPLETED') {
        onCompleteRef.current?.();
      }
    }
    // DO NOT include state.currentStage in deps - causes infinite loop
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStageId, stages]);

  // Track WebSocket events
  useEffect(() => {
    setState(prev => ({ ...prev, events: wsEvents }));
  }, [wsEvents]);

  // Track connection errors
  useEffect(() => {
    if (connectionStatus === 'error') {
      const error = 'WebSocket connection failed. Backend may not be running.';
      setState(prev => ({ ...prev, error }));
      onErrorRef.current?.(error);
    } else if (connectionStatus === 'open') {
      setState(prev => ({ ...prev, error: null }));
    }
  }, [connectionStatus]);

  // Start pipeline by calling backend API
  const startPipeline = useCallback(async (prompt: string) => {
    if (!apiBase) {
      const error = 'API base URL not configured';
      setState(prev => ({ ...prev, error }));
      onErrorRef.current?.(error);
      return;
    }

    setState(prev => ({ ...prev, isRunning: true, error: null }));

    try {
      // Step 1: Parse intent
      const parseResponse = await fetch(`${apiBase}/api/projects/${projectId}/parse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });

      if (!parseResponse.ok) {
        throw new Error(`Parse intent failed: ${parseResponse.statusText}`);
      }

      const parseResult = await parseResponse.json();
      console.log('[Pipeline] Intent parsed:', parseResult);

      // The backend will emit events via WebSocket as it progresses
      // We just need to trigger the start, the rest happens via WebSocket events

    } catch (err) {
      const error = err instanceof Error ? err.message : 'Pipeline start failed';
      setState(prev => ({ ...prev, error, isRunning: false }));
      onErrorRef.current?.(error);
      throw err;
    }
  }, [apiBase, projectId]);

  // Upload dataset to start data ingestion
  const uploadDataset = useCallback(async (file: File) => {
    if (!apiBase) {
      const error = 'API base URL not configured';
      setState(prev => ({ ...prev, error }));
      onErrorRef.current?.(error);
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${apiBase}/api/projects/${projectId}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('[Pipeline] Dataset uploaded:', result);

      return result;
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Upload failed';
      setState(prev => ({ ...prev, error }));
      onErrorRef.current?.(error);
      throw err;
    }
  }, [apiBase, projectId]);

  // Confirm current stage to advance pipeline
  const confirmStage = useCallback(async () => {
    if (!apiBase) {
      const error = 'API base URL not configured';
      setState(prev => ({ ...prev, error }));
      onErrorRef.current?.(error);
      return;
    }

    try {
      // Special handling for DATA_SOURCE stage - trigger download
      if (state.currentStage === 'DATA_SOURCE') {
        console.log('[Pipeline] 📥 Triggering dataset download...');
        const downloadResponse = await fetch(`${apiBase}/api/projects/${projectId}/dataset/download`, {
          method: 'POST',
        });

        if (!downloadResponse.ok) {
          const errorData = await downloadResponse.json();
          throw new Error(errorData.detail || `Download failed: ${downloadResponse.statusText}`);
        }

        const downloadResult = await downloadResponse.json();
        console.log('[Pipeline] ✅ Dataset downloaded:', downloadResult);
      }

      const response = await fetch(`${apiBase}/api/projects/${projectId}/confirm`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Confirm failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Check if backend rejected the confirmation
      if (result.error) {
        console.warn('[Pipeline] ❌ Confirmation rejected:', result.error);
        const error = result.error;
        setState(prev => ({ ...prev, error }));
        onErrorRef.current?.(error);
        throw new Error(error);
      }
      
      console.log('[Pipeline] ✅ Stage confirmed:', result);
      return result;
    } catch (err) {
      const error = err instanceof Error ? err.message : 'Confirm failed';
      setState(prev => ({ ...prev, error }));
      onErrorRef.current?.(error);
      throw err;
    }
  }, [apiBase, projectId, state.currentStage]);

  // Send chat message via WebSocket
  const sendChatMessage = useCallback((text: string) => {
    const wsClient = useProjectStore.getState().wsClient;
    if (!wsClient) {
      const error = 'WebSocket not connected';
      setState(prev => ({ ...prev, error }));
      onErrorRef.current?.(error);
      return;
    }

    wsClient.send({
      type: 'chat',
      text,
    });
  }, []);

  return {
    // State
    isRunning: state.isRunning,
    currentStage: state.currentStage,
    error: state.error,
    events: state.events,
    connectionStatus,
    stages,
    
    // Actions
    startPipeline,
    uploadDataset,
    confirmStage,
    sendChatMessage,
  };
}
