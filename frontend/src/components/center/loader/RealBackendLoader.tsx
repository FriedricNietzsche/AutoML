/**
 * Real Backend Training Pipeline Component
 * NO MOCKS - connects directly to backend WebSocket and REST APIs
 * 
 * If anything fails, it will show errors clearly instead of falling back to fake data
 */
import { useState, useEffect, useCallback } from 'react';
import { AlertCircle, CheckCircle2, Loader2, WifiOff, Sparkles } from 'lucide-react';
import { useBackendPipeline } from '../../../hooks/useBackendPipeline';
import type { EventEnvelope } from '../../../lib/ws';
import type { BuildSession } from '../../../lib/buildSession';

interface RealBackendLoaderProps {
  session: BuildSession | null;
  onComplete: () => void;
  updateFileContent: (path: string, content: string | ((prev: string) => string)) => void;
}

export default function RealBackendLoader({ session, onComplete, updateFileContent }: RealBackendLoaderProps) {
  const projectId = 'demo-project';
  const [errorDetails, setErrorDetails] = useState<string | null>(null);
  const [pipelineStarted, setPipelineStarted] = useState(false);
  const [isConfirming, setIsConfirming] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [uploadingFile, setUploadingFile] = useState(false);

  const {
    connectionStatus,
    currentStage,
    error,
    events,
    stages,
    startPipeline,
    confirmStage,
    sendChatMessage,
  } = useBackendPipeline({
    projectId,
    onStageChange: (stage) => {
      console.log('[RealBackendLoader] Stage changed:', stage);
      updateFileContent('/logs/training.log', (prev) => 
        `${prev || ''}[${new Date().toISOString()}] Stage changed: ${stage}\n`
      );
    },
    onError: (err) => {
      console.error('[RealBackendLoader] Error:', err);
      setErrorDetails(err);
      setIsConfirming(false); // Reset on error
    },
    onComplete: () => {
      console.log('[RealBackendLoader] Pipeline complete!');
      onComplete();
    },
  });

  // Check if we've actually started the pipeline (received PROMPT_PARSED or similar)
  const hasPipelineEvents = events.some(evt => 
    evt.event?.name === 'PROMPT_PARSED' || 
    evt.event?.name === 'MODEL_CANDIDATES' ||
    evt.event?.name === 'TRAIN_PROGRESS'
  );

  // Track all events and write to files
  useEffect(() => {
    if (events.length === 0) return;
    
    const latestEvent = events[events.length - 1];
    const eventName = latestEvent.event?.name || latestEvent.type || 'UNKNOWN';
    const payload = latestEvent.event?.payload || {};
    const payloadStr = JSON.stringify(payload).slice(0, 100);
    
    // Log event
    updateFileContent('/logs/training.log', (prev) => 
      `${prev || ''}[${new Date().toISOString()}] ${eventName}: ${payloadStr}\n`
    );

    // Handle specific events to write artifacts
    if (eventName === 'DATASET_SAMPLE_READY') {
      const assetUrl = (payload as Record<string, unknown>)?.asset_url;
      if (assetUrl) {
        updateFileContent('/artifacts/dataset_sample.json', JSON.stringify(payload, null, 2));
      }
    }

    if (eventName === 'PROFILE_SUMMARY') {
      updateFileContent('/artifacts/profile_summary.json', JSON.stringify(payload, null, 2));
    }

    if (eventName === 'EXPORT_READY') {
      updateFileContent('/artifacts/export_bundle.json', JSON.stringify(payload, null, 2));
    }
  }, [events, updateFileContent]);

  // Handle initial pipeline start - uses prompt from session
  const handleStart = useCallback(async () => {
    if (!session?.goalPrompt) {
      setErrorDetails('No goal prompt found in session');
      return;
    }
    
    try {
      setPipelineStarted(true);
      await startPipeline(session.goalPrompt);
    } catch (err) {
      console.error('[RealBackendLoader] Start failed:', err);
      setPipelineStarted(false);
    }
  }, [session, startPipeline]);

  // Handle dataset selection
  const handleSelectDataset = useCallback(async (datasetId: string) => {
    try {
      setSelectedDatasetId(datasetId);
      console.log('[RealBackendLoader] Selecting dataset:', datasetId);
      
      const response = await fetch(`http://localhost:8000/api/projects/${projectId}/dataset/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: datasetId }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to select dataset: ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('[RealBackendLoader] Dataset selected:', result);
      
      // Check if upload is required
      if (result.requires_upload) {
        setShowUploadDialog(true);
      }
      
    } catch (err) {
      console.error('[RealBackendLoader] Failed to select dataset:', err);
      setErrorDetails(err instanceof Error ? err.message : 'Failed to select dataset');
      setSelectedDatasetId(null);
    }
  }, [projectId]);

  // Handle file upload
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setUploadingFile(true);
      console.log('[RealBackendLoader] Uploading file:', file.name);

      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`http://localhost:8000/api/projects/${projectId}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to upload file: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('[RealBackendLoader] File uploaded:', result);
      
      setShowUploadDialog(false);
      setSelectedDatasetId('uploaded');
      
    } catch (err) {
      console.error('[RealBackendLoader] Upload failed:', err);
      setErrorDetails(err instanceof Error ? err.message : 'Failed to upload file');
    } finally {
      setUploadingFile(false);
    }
  }, [projectId]);

  // Handle model selection
  const handleSelectModel = useCallback(async (modelId: string) => {
    try {
      setSelectedModelId(modelId);
      console.log('[RealBackendLoader] Selecting model:', modelId);
      
      const response = await fetch(`http://localhost:8000/api/projects/${projectId}/model/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to select model: ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('[RealBackendLoader] Model selected:', result);
      
    } catch (err) {
      console.error('[RealBackendLoader] Failed to select model:', err);
      setErrorDetails(err instanceof Error ? err.message : 'Failed to select model');
      setSelectedModelId(null);
    }
  }, [projectId]);

  // Connection error - backend not available
  if (connectionStatus === 'error' || connectionStatus === 'closed') {
    return (
      <div className="h-full flex items-center justify-center bg-replit-bg p-8">
        <div className="max-w-2xl w-full bg-red-500/10 border border-red-500/30 rounded-lg p-8">
          <div className="flex items-start gap-4">
            <WifiOff className="w-8 h-8 text-red-500 flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-red-500 mb-4">Backend Connection Failed</h2>
              <p className="text-replit-text mb-4">
                Cannot connect to the backend WebSocket server. The backend may not be running.
              </p>
              
              <div className="bg-replit-surface/50 border border-replit-border rounded-lg p-4 mb-4 font-mono text-sm">
                <p className="text-replit-textMuted mb-2">Expected WebSocket URL:</p>
                <p className="text-yellow-400">ws://localhost:8000/ws/projects/{projectId}</p>
              </div>

              <div className="space-y-2 text-sm text-replit-textMuted mb-6">
                <p><strong className="text-replit-text">To fix this:</strong></p>
                <ol className="list-decimal list-inside space-y-1 ml-4">
                  <li>Open terminal in backend directory</li>
                  <li>Activate virtual environment: <code className="bg-replit-surface px-2 py-0.5 rounded">source .venv/bin/activate</code></li>
                  <li>Start server: <code className="bg-replit-surface px-2 py-0.5 rounded">uvicorn app.main:app --reload</code></li>
                  <li>Refresh this page</li>
                </ol>
              </div>

              <button
                onClick={() => window.location.reload()}
                className="px-6 py-3 bg-red-500 hover:bg-red-600 text-white rounded-lg font-medium transition-colors"
              >
                Retry Connection
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Connecting state
  if (connectionStatus === 'connecting' || connectionStatus === 'idle') {
    return (
      <div className="h-full flex items-center justify-center bg-replit-bg">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-replit-accent animate-spin mx-auto mb-4" />
          <p className="text-replit-textMuted">Connecting to backend...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error || errorDetails) {
    return (
      <div className="h-full flex items-center justify-center bg-replit-bg p-8">
        <div className="max-w-2xl w-full bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-8">
          <div className="flex items-start gap-4">
            <AlertCircle className="w-8 h-8 text-yellow-500 flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-yellow-500 mb-4">Pipeline Error</h2>
              <p className="text-replit-text mb-4">
                An error occurred while running the ML pipeline:
              </p>
              
              <div className="bg-replit-surface/50 border border-replit-border rounded-lg p-4 mb-6 font-mono text-sm text-red-400">
                {error || errorDetails}
              </div>

              <div className="flex gap-3">
                <button
                  onClick={() => {
                    setErrorDetails(null);
                    window.location.reload();
                  }}
                  className="px-6 py-3 bg-yellow-500 hover:bg-yellow-600 text-black rounded-lg font-medium transition-colors"
                >
                  Retry
                </button>
                <button
                  onClick={() => setErrorDetails(null)}
                  className="px-6 py-3 bg-replit-surface hover:bg-replit-surfaceHover border border-replit-border text-replit-text rounded-lg font-medium transition-colors"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Connected - show pipeline status
  return (
    <div className="h-full flex flex-col bg-replit-bg overflow-hidden">
      {/* Upload Dialog */}
      {showUploadDialog && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-replit-surface border border-replit-border rounded-lg p-8 max-w-md w-full mx-4">
            <h3 className="text-xl font-bold text-replit-text mb-4">Upload CSV Dataset</h3>
            <p className="text-sm text-replit-textMuted mb-6">
              Select a CSV file from your computer to use as your dataset.
            </p>
            
            <div className="border-2 border-dashed border-replit-border rounded-lg p-8 text-center mb-4 hover:border-blue-500/50 transition-colors">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                disabled={uploadingFile}
                className="hidden"
                id="csv-upload"
              />
              <label
                htmlFor="csv-upload"
                className="cursor-pointer flex flex-col items-center gap-3"
              >
                {uploadingFile ? (
                  <>
                    <Loader2 className="w-12 h-12 text-blue-500 animate-spin" />
                    <span className="text-replit-text">Uploading...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-12 h-12 text-replit-textMuted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <span className="text-replit-text font-medium">Click to browse</span>
                    <span className="text-xs text-replit-textMuted">or drag and drop your CSV file here</span>
                  </>
                )}
              </label>
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowUploadDialog(false);
                  setSelectedDatasetId(null);
                }}
                disabled={uploadingFile}
                className="flex-1 px-4 py-2 bg-replit-surface hover:bg-replit-surfaceHover border border-replit-border text-replit-text rounded-lg transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex-shrink-0 bg-replit-surface/60 border-b border-replit-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-replit-text">Real Backend Pipeline</h2>
            <p className="text-sm text-replit-textMuted mt-1">
              Connected to ws://localhost:8000 • No mock data
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            <span className="text-sm text-green-500 font-medium">Live</span>
          </div>
        </div>
      </div>

      {/* Current Stage - only show after pipeline starts */}
      {currentStage && (pipelineStarted || hasPipelineEvents) && (
        <div className="flex-shrink-0 bg-blue-500/10 border-b border-blue-500/30 px-6 py-4">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
            <div className="flex-1">
              <p className="text-sm text-replit-textMuted">Current Stage</p>
              <p className="text-lg font-semibold text-blue-500">{currentStage}</p>
              {stages[currentStage]?.status === 'IN_PROGRESS' && (
                <p className="text-xs text-yellow-400 mt-1">⏳ Processing... Please wait before confirming</p>
              )}
              {stages[currentStage]?.status === 'WAITING_CONFIRMATION' && (
                <p className="text-xs text-green-400 mt-1">✅ Ready to confirm and proceed</p>
              )}
            </div>
            <div className="ml-auto">
              <span className={`px-3 py-1 border rounded-full text-xs font-medium ${
                stages[currentStage]?.status === 'IN_PROGRESS' 
                  ? 'bg-yellow-500/20 border-yellow-500/30 text-yellow-400'
                  : stages[currentStage]?.status === 'WAITING_CONFIRMATION'
                  ? 'bg-green-500/20 border-green-500/30 text-green-400'
                  : 'bg-blue-500/20 border-blue-500/30 text-blue-400'
              }`}>
                {stages[currentStage]?.status || 'UNKNOWN'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Event Stream or Prompt Review */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="space-y-2">
          {!pipelineStarted && !hasPipelineEvents ? (
            // Prompt Review Screen (before pipeline starts)
            <div className="max-w-2xl mx-auto py-12">
              <div className="bg-replit-surface border border-replit-border rounded-lg p-8 shadow-sm">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                    <Sparkles className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-replit-text">Ready to Start</h3>
                    <p className="text-sm text-replit-textMuted">Review your configuration and begin</p>
                  </div>
                </div>

                {/* Goal Display */}
                <div className="mb-6">
                  <label className="text-xs font-semibold text-replit-textMuted uppercase tracking-wide mb-2 block">
                    Your Goal
                  </label>
                  <div className="bg-replit-bg border border-replit-border rounded-lg p-4">
                    <p className="text-replit-text leading-relaxed">
                      {session?.goalPrompt || 'No goal specified'}
                    </p>
                  </div>
                </div>

                {/* Dataset Display (if provided) */}
                {session?.datasetLinks && session.datasetLinks.length > 0 && (
                  <div className="mb-6">
                    <label className="text-xs font-semibold text-replit-textMuted uppercase tracking-wide mb-2 block">
                      Dataset Source{session.datasetLinks.length > 1 ? 's' : ''}
                    </label>
                    <div className="space-y-2">
                      {session.datasetLinks.map((link, idx) => (
                        <div key={idx} className="bg-replit-bg border border-replit-border rounded-lg p-3">
                          <p className="text-sm text-blue-400 font-mono truncate">{link}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Start Button */}
                <button
                  onClick={handleStart}
                  className="w-full px-6 py-4 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
                >
                  <Sparkles className="w-5 h-5" />
                  Start Pipeline
                </button>

                <p className="text-xs text-replit-textMuted text-center mt-4">
                  This will connect to the backend and begin real-time processing
                </p>
              </div>
            </div>
          ) : (
            // Event Stream (after pipeline starts)
            <>
              {!hasPipelineEvents && pipelineStarted && (
                <div className="text-center py-12">
                  <Loader2 className="w-12 h-12 text-replit-accent mx-auto mb-4 animate-spin" />
                  <p className="text-replit-textMuted">Waiting for backend response...</p>
                  <p className="text-xs text-replit-textMuted mt-2">Calling /parse endpoint...</p>
                </div>
              )}

          {events.map((evt, idx) => {
            const eventName = evt.event?.name || evt.type || 'UNKNOWN';
            const isError = eventName.includes('ERROR') || eventName.includes('FAILED');
            const isSuccess = eventName.includes('READY') || eventName.includes('COMPLETED');
            
            // Extract datasets array and message from payload
            const payload = evt.event?.payload;
            const datasetsArray = payload && typeof payload === 'object' 
              ? (payload as any).datasets 
              : null;
            const hasDatasets = Array.isArray(datasetsArray) && datasetsArray.length > 0;
            
            // Extract models array from payload
            const modelsArray = payload && typeof payload === 'object'
              ? (payload as any).models
              : null;
            const hasModels = Array.isArray(modelsArray) && modelsArray.length > 0;
            
            // Extract progress message from payload
            const progressMessage = payload && typeof payload === 'object'
              ? (payload as any).message
              : null;

            return (
              <div
                key={idx}
                className={`p-4 rounded-lg border ${
                  isError
                    ? 'bg-red-500/10 border-red-500/30'
                    : isSuccess
                    ? 'bg-green-500/10 border-green-500/30'
                    : 'bg-replit-surface/60 border-replit-border'
                }`}
              >
                <div className="flex items-start gap-3">
                  {isError ? (
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                  ) : isSuccess ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                  ) : (
                    <div className="w-5 h-5 flex-shrink-0 mt-0.5 flex items-center justify-center">
                      <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline gap-2 mb-1">
                      <span className={`font-semibold ${
                        isError ? 'text-red-400' : isSuccess ? 'text-green-400' : 'text-replit-text'
                      }`}>
                        {eventName}
                      </span>
                      <span className="text-xs text-replit-textMuted">
                        seq {evt.seq} • {new Date(evt.ts || 0).toLocaleTimeString()}
                      </span>
                    </div>
                    
                    {/* Display progress message if available */}
                    {progressMessage && (
                      <div className="mt-2 text-sm text-replit-text">
                        {progressMessage}
                      </div>
                    )}
                    
                    {payload && typeof payload === 'object' && (
                      <>
                        {/* Special rendering for DATASET_CANDIDATES with datasets */}
                        {eventName === 'DATASET_CANDIDATES' && hasDatasets ? (
                          <div className="mt-3 space-y-2">
                            <p className="text-sm text-replit-text mb-2">
                              Found {datasetsArray.length} dataset{datasetsArray.length !== 1 ? 's' : ''}. Click to select:
                            </p>
                            {datasetsArray.map((dataset: any, datasetIdx: number) => (
                              <button
                                key={datasetIdx}
                                onClick={() => handleSelectDataset(dataset.id || dataset.full_name)}
                                disabled={selectedDatasetId === (dataset.id || dataset.full_name)}
                                className={`w-full text-left p-3 rounded border transition-colors ${
                                  selectedDatasetId === (dataset.id || dataset.full_name)
                                    ? 'bg-blue-500/20 border-blue-500 cursor-default'
                                    : 'bg-replit-surface hover:bg-replit-surface/80 border-replit-border hover:border-blue-500/50'
                                }`}
                              >
                                <div className="flex items-start justify-between gap-2">
                                  <div className="flex-1">
                                    <div className="font-medium text-replit-text">
                                      {dataset.name || dataset.id || 'Unknown Dataset'}
                                    </div>
                                    {dataset.description && (
                                      <div className="text-xs text-replit-textMuted mt-1">
                                        {dataset.description}
                                      </div>
                                    )}
                                    <div className="flex items-center gap-3 mt-2">
                                      {dataset.downloads && (
                                        <div className="text-xs text-green-400">
                                          📥 {dataset.downloads.toLocaleString()} downloads
                                        </div>
                                      )}
                                      {dataset.url && (
                                        <a
                                          href={dataset.url}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          onClick={(e) => e.stopPropagation()}
                                          className="text-xs text-blue-400 hover:text-blue-300 underline flex items-center gap-1"
                                        >
                                          🔗 View on HuggingFace
                                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                          </svg>
                                        </a>
                                      )}
                                    </div>
                                  </div>
                                  {selectedDatasetId === (dataset.id || dataset.full_name) && (
                                    <div className="text-blue-400 flex-shrink-0">
                                      <CheckCircle2 className="w-5 h-5" />
                                    </div>
                                  )}
                                </div>
                              </button>
                            ))}
                          </div>
                        ) : eventName === 'MODEL_CANDIDATES' && hasModels ? (
                          /* Special rendering for MODEL_CANDIDATES with models */
                          <div className="mt-3 space-y-2">
                            <p className="text-sm text-replit-text mb-2">
                              Recommended {modelsArray.length} model{modelsArray.length !== 1 ? 's' : ''}. Click to select:
                            </p>
                            {modelsArray.map((model: any, modelIdx: number) => (
                              <button
                                key={modelIdx}
                                onClick={() => handleSelectModel(model.id)}
                                disabled={selectedModelId === model.id}
                                className={`w-full text-left p-3 rounded border transition-colors ${
                                  selectedModelId === model.id
                                    ? 'bg-green-500/20 border-green-500 cursor-default'
                                    : 'bg-replit-surface hover:bg-replit-surface/80 border-replit-border hover:border-green-500/50'
                                }`}
                              >
                                <div className="font-medium text-replit-text">
                                  {model.name || model.id || 'Unknown Model'}
                                </div>
                                {model.why && (
                                  <div className="text-xs text-replit-textMuted mt-1">
                                    {model.why}
                                  </div>
                                )}
                                {model.family && (
                                  <div className="text-xs text-purple-400 mt-1">
                                    Type: {model.family}
                                  </div>
                                )}
                                {selectedModelId === model.id && (
                                  <div className="text-xs text-green-400 mt-2">✓ Selected</div>
                                )}
                              </button>
                            ))}
                          </div>
                        ) : (
                          <pre className="text-xs text-replit-textMuted mt-2 overflow-x-auto">
                            {JSON.stringify(payload, null, 2)}
                          </pre>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
            </>
          )}
        </div>
      </div>

      {/* Actions */}
      {currentStage && stages[currentStage]?.status === 'WAITING_CONFIRMATION' && (
        <div className="flex-shrink-0 bg-replit-surface/60 border-t border-replit-border px-6 py-4">
          <div className="flex items-center gap-3">
            <button
              onClick={async () => {
                if (isConfirming) return;
                setIsConfirming(true);
                try {
                  await confirmStage();
                } finally {
                  setIsConfirming(false);
                }
              }}
              disabled={isConfirming || (currentStage === 'DATA_SOURCE' && !selectedDatasetId) || (currentStage === 'MODEL_SELECT' && !selectedModelId)}
              className="px-6 py-3 bg-replit-accent hover:bg-replit-accent/90 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isConfirming ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Confirming...
                </>
              ) : currentStage === 'DATA_SOURCE' && !selectedDatasetId ? (
                'Select a Dataset First'
              ) : currentStage === 'MODEL_SELECT' && !selectedModelId ? (
                'Select a Model First'
              ) : (
                'Confirm & Continue'
              )}
            </button>
            <input
              type="text"
              placeholder="Send message to AI agent..."
              className="flex-1 px-4 py-3 bg-replit-bg border border-replit-border rounded-lg text-replit-text placeholder-replit-textMuted focus:outline-none focus:ring-2 focus:ring-replit-accent"
              onKeyPress={(e) => {
                if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                  sendChatMessage(e.currentTarget.value);
                  e.currentTarget.value = '';
                }
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
