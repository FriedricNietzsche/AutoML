import { useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLiveMetrics } from '../hooks/useLiveMetrics';
import { useProjectStore } from '../stores/projectStore';

interface Stage {
  id: number;
  label: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: number;
  message?: string;
  artifacts?: Record<string, unknown>;
}

const STAGE_LABELS = [
  'Data Ingestion',
  'Data Profiling',
  'Preprocessing',
  'Model Training',
  'Review & Export',
];

export function TrainingLoader() {
  const projectId = useProjectStore((s) => s.projectId);
  const { events, metrics, isConnected, connectionError } = useLiveMetrics(projectId);

  // Derive stages from WS events
  const stages = useMemo<Stage[]>(() => {
    const stageMap = new Map<number, Stage>();
    
    // Initialize all stages as pending
    STAGE_LABELS.forEach((label, idx) => {
      stageMap.set(idx, {
        id: idx,
        label,
        status: 'pending',
        progress: 0,
      });
    });

    // Update from events
    events.forEach((event) => {
      if (event.type === 'STAGE_STATUS' && typeof event.uiBucket === 'number') {
        const bucket = event.uiBucket;
        const existing = stageMap.get(bucket);
        if (existing) {
          stageMap.set(bucket, {
            ...existing,
            status: event.status as Stage['status'],
            progress: event.progress ?? existing.progress,
            message: event.message,
            artifacts: event.artifacts,
          });
        }
      }
    });

    return Array.from(stageMap.values()).sort((a, b) => a.id - b.id);
  }, [events]);

  // Current active stage
  const activeStage = useMemo(() => {
    const running = stages.find((s) => s.status === 'running');
    if (running) return running;
    const lastCompleted = [...stages].reverse().find((s) => s.status === 'completed');
    return lastCompleted ?? stages[0];
  }, [stages]);

  // Latest metrics for display
  const latestMetrics = useMemo(() => {
    if (metrics.length === 0) return null;
    return metrics[metrics.length - 1];
  }, [metrics]);

  return (
    <div className="flex flex-col gap-6 p-6 bg-gray-900 rounded-xl">
      {/* Connection status */}
      <div className="flex items-center gap-2 text-sm">
        <div
          className={`w-2 h-2 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`}
        />
        <span className="text-gray-400">
          {isConnected ? 'Connected' : connectionError || 'Disconnected'}
        </span>
        <span className="text-gray-600 ml-auto text-xs">ID: {projectId}</span>
      </div>

      {/* Stage timeline */}
      <div className="flex items-center gap-2">
        {stages.map((stage, idx) => (
          <div key={stage.id} className="flex items-center">
            <StageIndicator stage={stage} isActive={stage.id === activeStage?.id} />
            {idx < stages.length - 1 && (
              <div
                className={`h-0.5 w-8 mx-1 ${
                  stage.status === 'completed' ? 'bg-green-500' : 'bg-gray-700'
                }`}
              />
            )}
          </div>
        ))}
      </div>

      {/* Active stage details */}
      <AnimatePresence mode="wait">
        {activeStage && (
          <motion.div
            key={activeStage.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-gray-800 rounded-lg p-4"
          >
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-white font-medium">{activeStage.label}</h3>
              <span className="text-sm text-gray-400">
                {Math.round(activeStage.progress * 100)}%
              </span>
            </div>
            
            {/* Progress bar */}
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden mb-2">
              <motion.div
                className="h-full bg-blue-500"
                initial={{ width: 0 }}
                animate={{ width: `${activeStage.progress * 100}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>

            {activeStage.message && (
              <p className="text-sm text-gray-400">{activeStage.message}</p>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Live metrics panel (during training) */}
      {activeStage?.id === 3 && latestMetrics && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h4 className="text-white text-sm font-medium mb-3">Training Metrics</h4>
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(latestMetrics).map(([key, value]) => (
              <div key={key} className="text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {typeof value === 'number' ? value.toFixed(4) : String(value)}
                </div>
                <div className="text-xs text-gray-500 uppercase">{key}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Artifacts preview */}
      {activeStage?.artifacts && Object.keys(activeStage.artifacts).length > 0 && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h4 className="text-white text-sm font-medium mb-2">Stage Output</h4>
          <pre className="text-xs text-gray-400 overflow-auto max-h-32">
            {JSON.stringify(activeStage.artifacts, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function StageIndicator({ stage, isActive }: { stage: Stage; isActive: boolean }) {
  const getStatusColor = () => {
    switch (stage.status) {
      case 'completed':
        return 'bg-green-500 text-white';
      case 'running':
        return 'bg-blue-500 text-white animate-pulse';
      case 'error':
        return 'bg-red-500 text-white';
      default:
        return 'bg-gray-700 text-gray-400';
    }
  };

  return (
    <div className="flex flex-col items-center">
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${getStatusColor()} ${
          isActive ? 'ring-2 ring-blue-400 ring-offset-2 ring-offset-gray-900' : ''
        }`}
      >
        {stage.status === 'completed' ? '✓' : stage.id + 1}
      </div>
      <span className="text-xs text-gray-500 mt-1 max-w-[60px] text-center truncate">
        {stage.label}
      </span>
    </div>
  );
}
