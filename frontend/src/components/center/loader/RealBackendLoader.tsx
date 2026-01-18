/**
 * Real Backend Training Pipeline Component
 * NO MOCKS - connects directly to backend WebSocket and REST APIs
 * 
 * If anything fails, it will show errors clearly instead of falling back to fake data
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { AlertCircle, CheckCircle2, Loader2, WifiOff, Sparkles, Brain, Database, Zap, Package, Rocket } from 'lucide-react';
import { useBackendPipeline } from '../../../hooks/useBackendPipeline';
import type { BuildSession } from '../../../lib/buildSession';
import confetti from 'canvas-confetti';

interface StageCardProps {
  icon: React.ElementType;
  title: string;
  stage: string;
  currentStage: string | null;
  stages: Record<string, { status: string }>;
  events: Array<{ event?: { name?: string; payload?: unknown }; type?: string; seq?: number; ts?: number }>;
  color: 'blue' | 'purple' | 'yellow' | 'green' | 'orange' | 'pink';
}

function StageCard({ icon: Icon, title, stage, currentStage, stages, events, color }: StageCardProps) {
  const stageStatus = stages[stage]?.status || 'PENDING';
  const isActive = currentStage === stage;
  const isCompleted = stageStatus === 'COMPLETED';
  const isInProgress = stageStatus === 'IN_PROGRESS' || (isActive && stageStatus === 'WAITING_CONFIRMATION');
  
  // Find relevant events for the current stage
  const stageEvents = events.filter(evt => {
    const name = evt.event?.name || evt.type || '';
    // Map event names to stages
    if (stage === 'PARSE_INTENT') return name.includes('PROMPT_PARSED') || name.includes('PARSE');
    if (stage === 'DATA_SOURCE') return name.includes('DATASET') && !name.includes('SAMPLE');
    if (stage === 'PROFILE_DATA' || stage === 'PROFILING' || stage === 'PREPROCESS') {
      return name.includes('PROFILE') || name.includes('SAMPLE') || name.includes('PREPROCESS');
    }
    if (stage === 'MODEL_SELECT') return name.includes('MODEL');
    if (stage === 'TRAIN' || stage === 'TRAINING') return name.includes('TRAIN');
    if (stage === 'EXPORT') return name.includes('EXPORT');
    return false;
  });

  // Extract key information from events
  let stageInfo: string | null = null;
  const latestEvent = stageEvents[stageEvents.length - 1];
  if (latestEvent?.event?.payload) {
    const payload = latestEvent.event.payload as Record<string, unknown>;
    
    if (stage === 'PARSE_INTENT' && payload.task_type) {
      stageInfo = `${payload.task_type}`;
    } else if (stage === 'DATA_SOURCE' && payload.datasets) {
      const datasets = payload.datasets as Array<unknown>;
      stageInfo = `${datasets.length} dataset${datasets.length !== 1 ? 's' : ''} found`;
    } else if (stage === 'PROFILING') {
      if (payload.profile) {
        const profile = payload.profile as Record<string, unknown>;
        const summary = profile.summary as Record<string, unknown> | undefined;
        if (summary?.missing_percentage !== undefined) {
          stageInfo = `${Number(summary.missing_percentage).toFixed(1)}% missing data`;
        } else {
          stageInfo = `${profile.rows || 0} rows, ${profile.columns || 0} cols`;
        }
      } else if (payload.steps) {
        const steps = payload.steps as Array<unknown>;
        stageInfo = `${steps.length} preprocessing step${steps.length !== 1 ? 's' : ''}`;
      }
    } else if (stage === 'MODEL_SELECT' && payload.models) {
      const models = payload.models as Array<unknown>;
      stageInfo = `${models.length} model${models.length !== 1 ? 's' : ''} available`;
    } else if (stage === 'TRAINING') {
      if (payload.progress !== undefined) {
        stageInfo = `${payload.progress}% complete`;
      } else if (payload.epoch && payload.total_epochs) {
        stageInfo = `Epoch ${payload.epoch}/${payload.total_epochs}`;
      } else if (payload.metrics) {
        stageInfo = 'Training finished';
      }
    } else if (stage === 'EXPORT' && payload.export) {
      stageInfo = 'Ready to download';
    }
  }

  const colorClasses = {
    blue: {
      bg: 'from-blue-500/20 to-blue-600/10',
      border: 'border-blue-500/50',
      activeBorder: 'border-blue-500',
      text: 'text-blue-400',
      icon: 'bg-blue-500',
      pulse: 'shadow-blue-500/50',
    },
    purple: {
      bg: 'from-purple-500/20 to-purple-600/10',
      border: 'border-purple-500/50',
      activeBorder: 'border-purple-500',
      text: 'text-purple-400',
      icon: 'bg-purple-500',
      pulse: 'shadow-purple-500/50',
    },
    yellow: {
      bg: 'from-yellow-500/20 to-yellow-600/10',
      border: 'border-yellow-500/50',
      activeBorder: 'border-yellow-500',
      text: 'text-yellow-400',
      icon: 'bg-yellow-500',
      pulse: 'shadow-yellow-500/50',
    },
    green: {
      bg: 'from-green-500/20 to-green-600/10',
      border: 'border-green-500/50',
      activeBorder: 'border-green-500',
      text: 'text-green-400',
      icon: 'bg-green-500',
      pulse: 'shadow-green-500/50',
    },
    orange: {
      bg: 'from-orange-500/20 to-orange-600/10',
      border: 'border-orange-500/50',
      activeBorder: 'border-orange-500',
      text: 'text-orange-400',
      icon: 'bg-orange-500',
      pulse: 'shadow-orange-500/50',
    },
    pink: {
      bg: 'from-pink-500/20 to-pink-600/10',
      border: 'border-pink-500/50',
      activeBorder: 'border-pink-500',
      text: 'text-pink-400',
      icon: 'bg-pink-500',
      pulse: 'shadow-pink-500/50',
    },
  };

  const colors = colorClasses[color];

  return (
    <div 
      className={`relative p-4 rounded-xl border-2 transition-all ${
        isActive 
          ? `bg-gradient-to-br ${colors.bg} ${colors.activeBorder} shadow-lg ${colors.pulse} animate-pulse-slow`
          : isCompleted
          ? `bg-gradient-to-br from-green-500/10 to-green-600/5 border-green-500/30`
          : `bg-replit-surface/40 ${colors.border} opacity-60`
      }`}
    >
      <div className="flex items-start gap-3">
        <div className={`w-10 h-10 rounded-lg ${
          isCompleted ? 'bg-green-500' : isActive ? colors.icon : 'bg-replit-textMuted'
        } flex items-center justify-center flex-shrink-0`}>
          {isCompleted ? (
            <CheckCircle2 className="w-6 h-6 text-white" />
          ) : isInProgress ? (
            <Loader2 className="w-6 h-6 text-white animate-spin" />
          ) : (
            <Icon className="w-6 h-6 text-white" />
          )}
        </div>
        
        <div className="flex-1 min-w-0">
          <h4 className={`font-bold mb-1 ${
            isCompleted ? 'text-green-400' : isActive ? colors.text : 'text-replit-textMuted'
          }`}>
            {title}
          </h4>
          
          <div className="text-xs mb-1">
            {isCompleted ? (
              <span className="text-green-400">✓ Completed</span>
            ) : isInProgress ? (
              <span className={colors.text}>In Progress...</span>
            ) : (
              <span className="text-replit-textMuted">Waiting...</span>
            )}
          </div>
          
          {stageInfo && (
            <div className={`text-xs font-medium ${
              isActive || isCompleted ? 'text-replit-text' : 'text-replit-textMuted'
            }`}>
              {stageInfo}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

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
  const [hasStageError, setHasStageError] = useState(false);
  const confettiTriggered = useRef(false);

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
      // Clear error flag on successful stage change
      setHasStageError(false);
    },
    onError: (err) => {
      console.error('[RealBackendLoader] Error:', err);
      setErrorDetails(err);
      setIsConfirming(false); // Reset on error
      setHasStageError(true); // Set error flag
    },
    onComplete: () => {
      console.log('[RealBackendLoader] Pipeline complete!');
      
      // Trigger confetti celebration!
      if (!confettiTriggered.current) {
        confettiTriggered.current = true;
        
        // Multiple confetti bursts for extra celebration
        const duration = 3000;
        const animationEnd = Date.now() + duration;
        const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 0 };

        function randomInRange(min: number, max: number) {
          return Math.random() * (max - min) + min;
        }

        const interval = setInterval(function() {
          const timeLeft = animationEnd - Date.now();

          if (timeLeft <= 0) {
            clearInterval(interval);
            return;
          }

          const particleCount = 50 * (timeLeft / duration);
          
          confetti({
            ...defaults,
            particleCount,
            origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 }
          });
          confetti({
            ...defaults,
            particleCount,
            origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 }
          });
        }, 250);
      }
      
      onComplete();
    },
  });

  // Track completed stages
  useEffect(() => {
    // Stage tracking for visual effects (optional)
    if (currentStage && stages[currentStage]?.status === 'COMPLETED') {
      console.log('[RealBackendLoader] Stage completed:', currentStage);
    }
  }, [currentStage, stages]);

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

  // Handle choosing a different dataset (reset selection)
  const handleChooseDifferentDataset = useCallback(() => {
    setSelectedDatasetId(null);
    setHasStageError(false);
  }, []);

  // Handle choosing a different model (reset selection)
  const handleChooseDifferentModel = useCallback(() => {
    setSelectedModelId(null);
    setHasStageError(false);
  }, []);

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

  // Render interactive content based on current stage
  const renderStageInteractiveContent = useCallback(() => {
    if (!currentStage) return null;

    // Find relevant events for the current stage
    const stageEvents = events.filter(evt => {
      const name = evt.event?.name || evt.type || '';
      if (currentStage === 'PARSE_INTENT') return name.includes('PROMPT_PARSED') || name.includes('PARSE');
      if (currentStage === 'DATA_SOURCE') return name.includes('DATASET');
      if (currentStage === 'PROFILE_DATA' || currentStage === 'PREPROCESS') {
        return name.includes('PROFILE') || name.includes('SAMPLE') || name.includes('PREPROCESS');
      }
      if (currentStage === 'MODEL_SELECT') return name.includes('MODEL');
      if (currentStage === 'TRAIN') return name.includes('TRAIN');
      if (currentStage === 'EXPORT') return name.includes('EXPORT');
      return false;
    });

    const latestEvent = stageEvents[stageEvents.length - 1];
    if (!latestEvent) return null;

    const eventName = latestEvent.event?.name || '';
    const payload = latestEvent.event?.payload as Record<string, unknown> | undefined;

    // Show parsed intent information
    if (eventName === 'PROMPT_PARSED' && payload) {
      return (
        <div className="bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-2 border-blue-500/50 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center flex-shrink-0">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h4 className="text-xl font-bold text-blue-400 mb-3">✓ Understood Your Request</h4>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {payload.task && (
                  <div className="bg-replit-surface/50 rounded-lg p-3">
                    <div className="text-xs font-semibold text-replit-textMuted uppercase mb-1">Task</div>
                    <div className="text-sm text-replit-text font-medium">{String(payload.task)}</div>
                  </div>
                )}
                
                {payload.task_type && (
                  <div className="bg-replit-surface/50 rounded-lg p-3">
                    <div className="text-xs font-semibold text-replit-textMuted uppercase mb-1">ML Type</div>
                    <div className="text-sm text-blue-400 font-bold uppercase">{String(payload.task_type)}</div>
                  </div>
                )}
                
                {payload.target && (
                  <div className="bg-replit-surface/50 rounded-lg p-3">
                    <div className="text-xs font-semibold text-replit-textMuted uppercase mb-1">Target Column</div>
                    <div className="text-sm text-green-400 font-mono">{String(payload.target)}</div>
                  </div>
                )}
                
                {payload.features && Array.isArray(payload.features) && (
                  <div className="bg-replit-surface/50 rounded-lg p-3">
                    <div className="text-xs font-semibold text-replit-textMuted uppercase mb-1">Features</div>
                    <div className="text-sm text-replit-text">{payload.features.length} columns identified</div>
                  </div>
                )}
              </div>
              
              {payload.summary && (
                <div className="mt-3 text-sm text-replit-textMuted">
                  {String(payload.summary)}
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }

    // Show profiling information - DETAILED
    if ((eventName === 'PROFILE_SUMMARY' || eventName === 'DATASET_SAMPLE_READY') && payload?.profile) {
      const profile = payload.profile as Record<string, unknown>;
      const summary = profile.summary as Record<string, unknown> | undefined;
      
      return (
        <div className="bg-gradient-to-br from-yellow-500/10 to-orange-500/10 border-2 border-yellow-500/50 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-4 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-yellow-500 to-orange-500 flex items-center justify-center flex-shrink-0">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h4 className="text-xl font-bold text-yellow-400 mb-2">📊 Dataset Analysis Complete</h4>
              <p className="text-sm text-replit-textMuted">
                We've analyzed your data in detail. Here's what we found:
              </p>
            </div>
          </div>

          {/* Main Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-replit-surface/50 rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-yellow-400">
                {profile.rows ? Number(profile.rows).toLocaleString() : 'N/A'}
              </div>
              <div className="text-xs text-replit-textMuted mt-1 uppercase">Total Rows</div>
            </div>
            
            <div className="bg-replit-surface/50 rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-blue-400">
                {profile.columns || 'N/A'}
              </div>
              <div className="text-xs text-replit-textMuted mt-1 uppercase">Columns</div>
            </div>
            
            <div className="bg-replit-surface/50 rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-green-400">
                {summary?.numeric_column_count || 0}
              </div>
              <div className="text-xs text-replit-textMuted mt-1 uppercase">Numeric</div>
            </div>
            
            <div className="bg-replit-surface/50 rounded-lg p-4 text-center">
              <div className="text-3xl font-bold text-purple-400">
                {summary?.categorical_column_count || 0}
              </div>
              <div className="text-xs text-replit-textMuted mt-1 uppercase">Categorical</div>
            </div>
          </div>

          {/* Data Quality Metrics */}
          {summary && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="bg-replit-surface/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-replit-text">Missing Data</span>
                  <span className={`text-sm font-bold ${
                    Number(summary.missing_percentage || 0) > 10 ? 'text-red-400' : 
                    Number(summary.missing_percentage || 0) > 5 ? 'text-yellow-400' : 'text-green-400'
                  }`}>
                    {Number(summary.missing_percentage || 0).toFixed(2)}%
                  </span>
                </div>
                <div className="h-2 bg-replit-bg rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all ${
                      Number(summary.missing_percentage || 0) > 10 ? 'bg-red-500' : 
                      Number(summary.missing_percentage || 0) > 5 ? 'bg-yellow-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${Math.min(Number(summary.missing_percentage || 0), 100)}%` }}
                  />
                </div>
              </div>

              <div className="bg-replit-surface/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-replit-text">Memory Usage</span>
                  <span className="text-sm font-bold text-blue-400">
                    {summary.memory_usage_mb ? `${Number(summary.memory_usage_mb).toFixed(2)} MB` : 'N/A'}
                  </span>
                </div>
                <div className="text-xs text-replit-textMuted">
                  {summary.total_cells ? `${Number(summary.total_cells).toLocaleString()} total cells` : ''}
                </div>
              </div>
            </div>
          )}

          {/* Column Details */}
          {profile.column_types && (
            <div className="bg-replit-surface/50 rounded-lg p-4">
              <div className="text-sm font-semibold text-replit-text mb-3">📋 Column Details</div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-48 overflow-y-auto">
                {Object.entries(profile.column_types as Record<string, string>).map(([col, type]) => (
                  <div key={col} className="flex items-center gap-2 text-xs p-2 bg-replit-bg rounded">
                    <span className={`px-2 py-0.5 rounded font-mono font-semibold ${
                      type === 'int64' || type === 'float64' ? 'bg-green-500/20 text-green-400' :
                      type === 'object' || type === 'category' ? 'bg-purple-500/20 text-purple-400' :
                      type === 'bool' ? 'bg-blue-500/20 text-blue-400' :
                      'bg-gray-500/20 text-gray-400'
                    }`}>
                      {String(type)}
                    </span>
                    <span className="text-replit-text truncate">{col}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Statistical Summary */}
          {profile.stats && (
            <div className="mt-4 bg-replit-surface/50 rounded-lg p-4">
              <div className="text-sm font-semibold text-replit-text mb-3">📈 Statistical Summary</div>
              <div className="text-xs text-replit-textMuted space-y-1">
                <div>• Data completeness: {summary?.completeness_percentage ? `${Number(summary.completeness_percentage).toFixed(1)}%` : 'N/A'}</div>
                {summary?.duplicate_rows !== undefined && (
                  <div className={Number(summary.duplicate_rows) > 0 ? 'text-yellow-400' : ''}>
                    • Duplicate rows: {String(summary.duplicate_rows)}
                  </div>
                )}
                {summary?.columns_with_missing && (
                  <div>• Columns with missing values: {String(summary.columns_with_missing)}</div>
                )}
              </div>
            </div>
          )}
        </div>
      );
    }

    // Show preprocessing information - DETAILED
    if (eventName === 'PREPROCESS_COMPLETE' && payload) {
      const steps = payload.steps as Array<Record<string, unknown>> | undefined;
      const transformations = payload.transformations as Record<string, unknown> | undefined;
      
      return (
        <div className="bg-gradient-to-br from-orange-500/10 to-red-500/10 border-2 border-orange-500/50 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-4 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center flex-shrink-0">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h4 className="text-xl font-bold text-orange-400 mb-2">🔧 Data Preprocessing Complete</h4>
              <p className="text-sm text-replit-textMuted">
                Your data has been cleaned and prepared for training. Here's what we did:
              </p>
            </div>
          </div>

          {/* Preprocessing Steps */}
          {steps && steps.length > 0 && (
            <div className="space-y-3 mb-4">
              {steps.map((step, idx) => (
                <div key={idx} className="bg-replit-surface/50 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-orange-500/20 text-orange-400 flex items-center justify-center font-bold text-sm flex-shrink-0">
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <div className="font-semibold text-replit-text mb-1">
                        {String(step.name || step.type || 'Transformation')}
                      </div>
                      {step.description && (
                        <div className="text-sm text-replit-textMuted mb-2">
                          {String(step.description)}
                        </div>
                      )}
                      {step.columns && Array.isArray(step.columns) && (
                        <div className="flex flex-wrap gap-1">
                          {step.columns.map((col, colIdx) => (
                            <span key={colIdx} className="text-xs px-2 py-1 bg-replit-bg text-blue-400 rounded font-mono">
                              {String(col)}
                            </span>
                          ))}
                        </div>
                      )}
                      {step.details && (
                        <div className="mt-2 text-xs text-replit-textMuted">
                          {String(step.details)}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Transformation Summary */}
          {transformations && (
            <div className="bg-replit-surface/50 rounded-lg p-4">
              <div className="text-sm font-semibold text-replit-text mb-3">📊 Transformation Summary</div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {transformations.encoded_columns && Array.isArray(transformations.encoded_columns) && (
                  <div>
                    <div className="text-xs text-replit-textMuted mb-2">Encoded Columns</div>
                    <div className="text-lg font-bold text-purple-400">
                      {transformations.encoded_columns.length}
                    </div>
                  </div>
                )}
                {transformations.scaled_columns && Array.isArray(transformations.scaled_columns) && (
                  <div>
                    <div className="text-xs text-replit-textMuted mb-2">Scaled Columns</div>
                    <div className="text-lg font-bold text-green-400">
                      {transformations.scaled_columns.length}
                    </div>
                  </div>
                )}
                {transformations.imputed_columns && Array.isArray(transformations.imputed_columns) && (
                  <div>
                    <div className="text-xs text-replit-textMuted mb-2">Imputed Columns</div>
                    <div className="text-lg font-bold text-yellow-400">
                      {transformations.imputed_columns.length}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Final Dataset Info */}
          {payload.final_shape && (
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3 text-center">
                <div className="text-xs text-replit-textMuted mb-1">Final Dataset Size</div>
                <div className="text-lg font-bold text-green-400">
                  {String(payload.final_shape)}
                </div>
              </div>
              {payload.train_test_split && (
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3 text-center">
                  <div className="text-xs text-replit-textMuted mb-1">Train/Test Split</div>
                  <div className="text-lg font-bold text-blue-400">
                    {String(payload.train_test_split)}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      );
    }

    // Show training progress - ANIMATED AND DETAILED
    if ((eventName === 'TRAIN_PROGRESS' || eventName === 'TRAIN_COMPLETE') && payload) {
      const progress = payload.progress as number | undefined;
      const metrics = payload.metrics as Record<string, unknown> | undefined;
      const epoch = payload.epoch as number | undefined;
      const totalEpochs = payload.total_epochs as number | undefined;
      const currentMetrics = payload.current_metrics as Record<string, unknown> | undefined;
      const isComplete = eventName === 'TRAIN_COMPLETE';
      
      return (
        <div className={`bg-gradient-to-br border-2 rounded-lg p-6 mb-6 transition-all ${
          isComplete 
            ? 'from-green-500/10 to-blue-500/10 border-green-500/50' 
            : 'from-orange-500/10 to-yellow-500/10 border-orange-500/50 animate-pulse-slow'
        }`}>
          <div className="flex items-start gap-4 mb-4">
            <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${
              isComplete 
                ? 'bg-gradient-to-br from-green-500 to-blue-500' 
                : 'bg-gradient-to-br from-orange-500 to-yellow-500 animate-pulse'
            }`}>
              {isComplete ? (
                <CheckCircle2 className="w-6 h-6 text-white" />
              ) : (
                <Rocket className="w-6 h-6 text-white" />
              )}
            </div>
            <div className="flex-1">
              <h4 className={`text-xl font-bold mb-2 ${isComplete ? 'text-green-400' : 'text-orange-400'}`}>
                {isComplete ? '✅ Training Complete!' : '🚀 Training in Progress'}
              </h4>
              <p className="text-sm text-replit-textMuted">
                {isComplete 
                  ? 'Your model has been successfully trained and is ready to use.'
                  : 'The model is learning from your data. This may take a few moments...'}
              </p>
            </div>
          </div>

          {/* Progress Bar - Large and Animated */}
          {!isComplete && progress !== undefined && (
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-replit-text">Overall Progress</span>
                <span className="text-2xl font-bold text-orange-400">{progress}%</span>
              </div>
              <div className="h-4 bg-replit-bg rounded-full overflow-hidden border border-orange-500/30">
                <div 
                  className="h-full bg-gradient-to-r from-orange-500 via-yellow-500 to-orange-500 transition-all duration-500 relative overflow-hidden"
                  style={{ width: `${progress}%` }}
                >
                  {/* Animated shine effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" 
                       style={{ 
                         animation: 'shimmer 2s infinite',
                         backgroundSize: '200% 100%'
                       }} 
                  />
                </div>
              </div>
            </div>
          )}

          {/* Epoch Progress */}
          {!isComplete && epoch !== undefined && totalEpochs !== undefined && (
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-semibold text-replit-text">Training Epoch</span>
                <span className="text-lg font-bold text-yellow-400">
                  {epoch} / {totalEpochs}
                </span>
              </div>
              <div className="h-2 bg-replit-bg rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-yellow-500 to-orange-500 transition-all duration-300"
                  style={{ width: `${(epoch / totalEpochs) * 100}%` }}
                />
              </div>
            </div>
          )}

          {/* Current Training Metrics (live updates) */}
          {!isComplete && currentMetrics && Object.keys(currentMetrics).length > 0 && (
            <div className="mb-6">
              <div className="text-sm font-semibold text-replit-text mb-3">📊 Live Metrics</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {Object.entries(currentMetrics).map(([key, value]) => (
                  <div key={key} className="bg-replit-surface/50 rounded-lg p-3 text-center">
                    <div className="text-xs text-replit-textMuted uppercase mb-1">
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div className="text-lg font-bold text-orange-400">
                      {typeof value === 'number' 
                        ? value < 1 
                          ? (value * 100).toFixed(2) + '%'
                          : value.toFixed(4)
                        : String(value)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Training Message */}
          {!isComplete && payload.message && (
            <div className="bg-replit-surface/30 border border-orange-500/30 rounded-lg p-3 mb-4">
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 text-orange-400 animate-spin" />
                <span className="text-sm text-replit-text">{String(payload.message)}</span>
              </div>
            </div>
          )}

          {/* Final Metrics (after training complete) */}
          {isComplete && metrics && (
            <div className="space-y-4">
              <div className="text-sm font-semibold text-replit-text mb-3">🎯 Final Performance Metrics</div>
              
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {Object.entries(metrics)
                  .filter(([key]) => !['model_name', 'task_type', 'dataset', 'timestamp'].includes(key))
                  .map(([key, value]) => {
                    const numValue = typeof value === 'number' ? value : parseFloat(String(value));
                    const isPercentage = numValue < 1 && numValue > 0;
                    const displayValue = isPercentage 
                      ? `${(numValue * 100).toFixed(2)}%`
                      : typeof value === 'number'
                      ? value.toFixed(4)
                      : String(value);
                    
                    // Color code based on metric name
                    const colorClass = 
                      key.includes('accuracy') || key.includes('score') ? 'text-green-400' :
                      key.includes('loss') || key.includes('error') ? 'text-red-400' :
                      key.includes('precision') ? 'text-blue-400' :
                      key.includes('recall') ? 'text-purple-400' :
                      key.includes('f1') ? 'text-yellow-400' :
                      'text-replit-text';
                    
                    return (
                      <div key={key} className="bg-replit-surface/50 rounded-lg p-4 text-center border border-green-500/20">
                        <div className="text-xs text-replit-textMuted uppercase mb-2">
                          {key.replace(/_/g, ' ')}
                        </div>
                        <div className={`text-2xl font-bold ${colorClass}`}>
                          {displayValue}
                        </div>
                      </div>
                    );
                  })}
              </div>

              {/* Model Info */}
              {metrics.model_name && (
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mt-4">
                  <div className="flex items-center gap-2">
                    <Brain className="w-5 h-5 text-blue-400" />
                    <div className="flex-1">
                      <div className="text-xs text-replit-textMuted">Trained Model</div>
                      <div className="text-sm font-semibold text-blue-400">{String(metrics.model_name)}</div>
                    </div>
                    {metrics.task_type && (
                      <div className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs font-semibold">
                        {String(metrics.task_type)}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Success Message */}
              <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 text-center">
                <div className="text-2xl mb-2">🎉</div>
                <div className="text-sm font-semibold text-green-400">
                  Training completed successfully! Your model is now ready for deployment.
                </div>
              </div>
            </div>
          )}
        </div>
      );
    }

    // Dataset selection
    if (eventName === 'DATASET_CANDIDATES' && payload?.datasets) {
      const datasets = payload.datasets as Array<{
        id: string;
        name?: string;
        full_name?: string;
        description?: string;
        downloads?: number;
        url?: string;
      }>;

      return (
        <div className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border-2 border-purple-500/50 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-4 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
              <Database className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h4 className="text-xl font-bold text-purple-400 mb-2">📊 Select a Dataset</h4>
              <p className="text-sm text-replit-textMuted">
                Found {datasets.length} dataset{datasets.length !== 1 ? 's' : ''}. Choose the one that best fits your needs.
              </p>
            </div>
          </div>
          
          <div className="space-y-3">
            {datasets.map((dataset, idx) => (
              <button
                key={idx}
                onClick={() => handleSelectDataset(dataset.id || dataset.full_name || '')}
                disabled={selectedDatasetId === (dataset.id || dataset.full_name)}
                className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                  selectedDatasetId === (dataset.id || dataset.full_name)
                    ? 'bg-blue-500/20 border-blue-500 cursor-default'
                    : 'bg-replit-surface/80 hover:bg-replit-surface border-replit-border hover:border-blue-500/50'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <div className="font-medium text-replit-text">
                      {dataset.name || dataset.id || 'Unknown Dataset'}
                    </div>
                    {dataset.description && (
                      <div className="text-sm text-replit-textMuted mt-1">
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
                          className="text-xs text-blue-400 hover:text-blue-300 underline"
                        >
                          🔗 View on HuggingFace
                        </a>
                      )}
                    </div>
                  </div>
                  {selectedDatasetId === (dataset.id || dataset.full_name) && (
                    <CheckCircle2 className="w-5 h-5 text-blue-400" />
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      );
    }

    // Model selection
    if (eventName === 'MODEL_CANDIDATES' && payload?.models) {
      const models = payload.models as Array<{
        id: string;
        name?: string;
        why?: string;
        family?: string;
      }>;

      return (
        <div className="bg-gradient-to-br from-green-500/10 to-blue-500/10 border-2 border-green-500/50 rounded-lg p-6 mb-6">
          <div className="flex items-start gap-4 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-green-500 to-blue-500 flex items-center justify-center flex-shrink-0">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h4 className="text-xl font-bold text-green-400 mb-2">🤖 Select Your Model</h4>
              <p className="text-sm text-replit-textMuted">
                We've analyzed your task and found {models.length} suitable model{models.length !== 1 ? 's' : ''}. The top choice is highlighted.
              </p>
            </div>
          </div>
          
          <div className="space-y-4">
            {/* Recommended (first) */}
            {models[0] && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm font-semibold text-green-400 bg-green-400/10 px-3 py-1 rounded-full">⭐ BEST MATCH FOR YOUR TASK</span>
                </div>
                <button
                  onClick={() => handleSelectModel(models[0].id)}
                  disabled={selectedModelId === models[0].id}
                  className={`w-full text-left p-5 rounded-xl border-2 transition-all ${
                    selectedModelId === models[0].id
                      ? 'bg-green-500/20 border-green-500 shadow-lg shadow-green-500/20'
                      : 'bg-gradient-to-br from-green-500/10 to-blue-500/10 border-green-500/50 hover:border-green-500 hover:shadow-lg'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="text-xl font-bold text-replit-text mb-2">
                        {models[0].name || models[0].id}
                      </div>
                      {models[0].family && (
                        <div className="inline-block px-3 py-1 text-xs text-purple-400 bg-purple-400/10 rounded-full mb-2">
                          {models[0].family}
                        </div>
                      )}
                      {models[0].why && (
                        <div className="text-sm text-replit-textMuted mt-2 leading-relaxed">
                          💡 <strong>Why this model:</strong> {models[0].why}
                        </div>
                      )}
                    </div>
                    {selectedModelId === models[0].id && (
                      <CheckCircle2 className="w-6 h-6 text-green-400" />
                    )}
                  </div>
                </button>
              </div>
            )}

            {/* Other options */}
            {models.length > 1 && (
              <div className="space-y-2">
                <p className="text-xs text-replit-textMuted">Other options:</p>
                {models.slice(1).map((model, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSelectModel(model.id)}
                    disabled={selectedModelId === model.id}
                    className={`w-full text-left p-3 rounded-lg border transition-all ${
                      selectedModelId === model.id
                        ? 'bg-green-500/20 border-green-500'
                        : 'bg-replit-surface/80 hover:bg-replit-surface border-replit-border hover:border-blue-500/50'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="font-medium text-sm text-replit-text">
                          {model.name || model.id}
                        </div>
                        {model.why && (
                          <div className="text-xs text-replit-textMuted mt-1">
                            {model.why}
                          </div>
                        )}
                      </div>
                      {selectedModelId === model.id && (
                        <CheckCircle2 className="w-5 h-5 text-green-400" />
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      );
    }

    // Export package
    if (eventName === 'EXPORT_READY' && payload?.export) {
      const exportData = payload.export as {
        download_url?: string;
        zip_filename?: string;
        files?: Record<string, string>;
        size_mb?: number;
      };

      return (
        <div className="bg-gradient-to-br from-green-500/10 to-blue-500/10 border-2 border-green-500/50 rounded-lg p-6">
          <div className="flex items-start gap-4">
            <div className="text-4xl">📦</div>
            <div className="flex-1">
              <h4 className="text-xl font-bold text-green-400 mb-2">Export Package Ready!</h4>
              <p className="text-sm text-replit-textMuted mb-4">
                Your trained model has been packaged and is ready to download.
              </p>
              
              <a
                href={`http://localhost:8000${exportData.download_url}`}
                download
                className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download {exportData.zip_filename || 'Export Package'}
              </a>
            </div>
          </div>
        </div>
      );
    }

    return null;
  }, [currentStage, events, selectedDatasetId, selectedModelId, handleSelectDataset, handleSelectModel]);

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
          // Stage-based visualization
          <div className="max-w-6xl mx-auto py-8 space-y-6">
            {/* Interactive Content at the TOP - User choices go here */}
            {renderStageInteractiveContent()}

            {/* Stage Progress Visualization */}
            <div>
              <h3 className="text-2xl font-bold text-replit-text mb-6 text-center">Pipeline Stages</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Stage 1: Understanding */}
                <StageCard
                  icon={Brain}
                  title="Understanding"
                  stage="PARSE_INTENT"
                  currentStage={currentStage}
                  stages={stages}
                  events={events}
                  color="blue"
                />

                {/* Stage 2: Data Source */}
                <StageCard
                  icon={Database}
                  title="Data Source"
                  stage="DATA_SOURCE"
                  currentStage={currentStage}
                  stages={stages}
                  events={events}
                  color="purple"
                />

                {/* Stage 3: Data Profiling */}
                <StageCard
                  icon={Zap}
                  title="Data Profiling"
                  stage="PROFILE_DATA"
                  currentStage={currentStage}
                  stages={stages}
                  events={events}
                  color="yellow"
                />

                {/* Stage 4: Model Selection */}
                <StageCard
                  icon={Brain}
                  title="Model Selection"
                  stage="MODEL_SELECT"
                  currentStage={currentStage}
                  stages={stages}
                  events={events}
                  color="green"
                />

                {/* Stage 5: Training */}
                <StageCard
                  icon={Rocket}
                  title="Training"
                  stage="TRAIN"
                  currentStage={currentStage}
                  stages={stages}
                  events={events}
                  color="orange"
                />

                {/* Stage 6: Export */}
                <StageCard
                  icon={Package}
                  title="Export"
                  stage="EXPORT"
                  currentStage={currentStage}
                  stages={stages}
                  events={events}
                  color="pink"
                />
              </div>
            </div>

            {/* Detailed Event Stream for Current Stage */}
            <div className="bg-replit-surface/60 border border-replit-border rounded-lg p-6">
              <h4 className="text-lg font-bold text-replit-text mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-blue-500" />
                Recent Events
              </h4>
              <div className="space-y-3 max-h-[400px] overflow-y-auto"
                style={{
                  scrollbarWidth: 'thin',
                  scrollbarColor: '#4a5568 #1a1b26'
                }}
              >
                {events.length === 0 && (
                  <div className="text-center py-12">
                    <Loader2 className="w-12 h-12 text-replit-accent mx-auto mb-4 animate-spin" />
                    <p className="text-replit-textMuted">Waiting for backend response...</p>
                  </div>
                )}
                
                {events.slice(-10).reverse().map((evt) => {
                  const eventName = evt.event?.name || evt.type || 'UNKNOWN';
                  const isError = eventName.includes('ERROR') || eventName.includes('FAILED');
                  const isSuccess = eventName.includes('READY') || eventName.includes('COMPLETED');
                  const payload = evt.event?.payload;
                  
                  return (
                    <div
                      key={evt.seq || Math.random()}
                      className={`p-3 rounded-lg border text-sm ${
                        isError
                          ? 'bg-red-500/10 border-red-500/30'
                          : isSuccess
                          ? 'bg-green-500/10 border-green-500/30'
                          : 'bg-replit-surface border-replit-border'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        {isError ? (
                          <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                        ) : isSuccess ? (
                          <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                        ) : (
                          <div className="w-4 h-4 flex-shrink-0 mt-0.5 flex items-center justify-center">
                            <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                          </div>
                        )}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-baseline gap-2">
                            <span className={`font-semibold text-xs ${
                              isError ? 'text-red-400' : isSuccess ? 'text-green-400' : 'text-replit-text'
                            }`}>
                              {eventName}
                            </span>
                            <span className="text-xs text-replit-textMuted">
                              {new Date(evt.ts || 0).toLocaleTimeString()}
                            </span>
                          </div>
                          {payload && typeof payload === 'object' && (payload as Record<string, unknown>).message && (
                            <p className="text-xs text-replit-textMuted mt-1">
                              {String((payload as Record<string, unknown>).message)}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Actions - Always visible */}
      {currentStage && (
        <div className="flex-shrink-0 bg-replit-surface/60 border-t border-replit-border px-6 py-4">
          <div className="flex flex-col gap-3">
            {/* Error message with option to go back */}
            {hasStageError && stages[currentStage]?.status === 'WAITING_CONFIRMATION' && (
              <div className="flex items-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <span className="text-sm text-red-400 flex-1">
                  {currentStage === 'DATA_SOURCE' 
                    ? 'This dataset cannot be loaded. Please choose a different dataset or upload a CSV file.'
                    : 'An error occurred. Please try a different selection.'}
                </span>
              </div>
            )}
            
            <div className="flex items-center gap-3">
              {/* Go Back / Choose Different button - only when waiting for confirmation */}
              {stages[currentStage]?.status === 'WAITING_CONFIRMATION' && (selectedDatasetId || selectedModelId) && (
                <button
                  onClick={() => {
                    if (currentStage === 'DATA_SOURCE') {
                      handleChooseDifferentDataset();
                    } else if (currentStage === 'MODEL_SELECT') {
                      handleChooseDifferentModel();
                    }
                  }}
                  className="px-4 py-3 bg-replit-surface hover:bg-replit-surface/80 border border-replit-border hover:border-yellow-500/50 text-replit-text rounded-lg font-medium transition-colors flex items-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                  </svg>
                  {currentStage === 'DATA_SOURCE' ? 'Choose Different Dataset' : 'Choose Different Model'}
                </button>
              )}
              
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
                disabled={isConfirming}
                className="px-6 py-3 bg-replit-accent hover:bg-replit-accent/90 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isConfirming ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Confirming...
                  </>
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
        </div>
      )}
    </div>
  );
}
