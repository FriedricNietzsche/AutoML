import { useTrainingStore, useLossCurve, useTrainingProgress } from '../../store/trainingStore';
import { Activity, TrendingDown, Clock, Zap } from 'lucide-react';
import { useMemo } from 'react';

// ============================================================================
// TRAINING PROGRESS BAR COMPONENT
// ============================================================================

export function TrainingProgressBar() {
  const { status, progress, isTraining, percentComplete } = useTrainingProgress();
  
  if (status === 'idle') {
    return null;
  }
  
  const statusConfig = {
    idle: { color: 'bg-replit-textMuted', label: 'Idle' },
    initializing: { color: 'bg-yellow-500', label: 'Initializing...' },
    training: { color: 'bg-replit-accent', label: 'Training' },
    evaluating: { color: 'bg-purple-500', label: 'Evaluating' },
    completed: { color: 'bg-replit-success', label: 'Complete' },
    failed: { color: 'bg-red-500', label: 'Failed' },
  };
  
  const config = statusConfig[status];
  
  return (
    <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Activity className={`w-4 h-4 ${isTraining ? 'animate-pulse' : ''}`} />
          <span className="text-sm font-medium text-replit-text">{config.label}</span>
        </div>
        <span className="text-sm text-replit-textMuted">{percentComplete.toFixed(1)}%</span>
      </div>
      
      {/* Progress bar */}
      <div className="h-2 bg-replit-bg rounded-full overflow-hidden">
        <div 
          className={`h-full ${config.color} transition-all duration-300 ease-out`}
          style={{ width: `${percentComplete}%` }}
        />
      </div>
      
      {/* Progress details */}
      {progress && (
        <div className="flex items-center justify-between mt-3 text-xs text-replit-textMuted">
          <span>
            Epoch {progress.epoch}/{progress.totalEpochs} • Step {progress.step}/{progress.totalSteps}
          </span>
          {progress.etaSeconds !== null && (
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              ETA: {progress.etaSeconds.toFixed(1)}s
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// LOSS CURVE COMPONENT (CSS-ONLY CHART)
// ============================================================================

export function LossCurve() {
  const { data, latest, min, max, hasData } = useLossCurve();
  const { status } = useTrainingProgress();
  
  // Normalize data for display
  const normalizedData = useMemo(() => {
    if (!hasData || data.length === 0) return [];
    
    const values = data.map(d => d.train).filter((v): v is number => v !== undefined);
    if (values.length === 0) return [];
    
    const dataMin = Math.min(...values);
    const dataMax = Math.max(...values);
    const range = dataMax - dataMin || 1;
    
    return data.map((point, index) => ({
      x: (index / (data.length - 1 || 1)) * 100,
      y: point.train !== undefined 
        ? ((point.train - dataMin) / range) * 100 
        : null,
      step: point.step,
      value: point.train,
    }));
  }, [data, hasData]);
  
  if (!hasData) {
    return (
      <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
        <h3 className="text-sm font-semibold text-replit-text mb-4 flex items-center gap-2">
          <TrendingDown className="w-4 h-4" />
          Loss Curve
        </h3>
        <div className="h-48 flex items-center justify-center text-replit-textMuted text-sm">
          {status === 'idle' ? 'Start training to see loss curve' : 'Waiting for data...'}
        </div>
      </div>
    );
  }
  
  // Build SVG path for the line
  const pathD = normalizedData
    .filter(p => p.y !== null)
    .map((point, i) => {
      const x = point.x;
      const y = 100 - (point.y ?? 0); // Invert Y (SVG 0 is top)
      return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(' ');
  
  return (
    <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-replit-text flex items-center gap-2">
          <TrendingDown className="w-4 h-4" />
          Loss Curve
        </h3>
        {latest !== null && (
          <div className="text-right">
            <span className="text-lg font-bold text-replit-accent">{latest.toFixed(4)}</span>
            <span className="text-xs text-replit-textMuted ml-1">current</span>
          </div>
        )}
      </div>
      
      {/* SVG Chart */}
      <div className="relative h-48 w-full">
        {/* Background grid */}
        <svg 
          className="absolute inset-0 w-full h-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          {/* Horizontal grid lines */}
          {[0, 25, 50, 75, 100].map(y => (
            <line 
              key={y}
              x1="0" y1={y} x2="100" y2={y}
              stroke="currentColor"
              strokeOpacity="0.1"
              strokeDasharray="2 2"
              className="text-replit-border"
            />
          ))}
          {/* Vertical grid lines */}
          {[0, 25, 50, 75, 100].map(x => (
            <line 
              key={x}
              x1={x} y1="0" x2={x} y2="100"
              stroke="currentColor"
              strokeOpacity="0.1"
              strokeDasharray="2 2"
              className="text-replit-border"
            />
          ))}
        </svg>
        
        {/* Loss line */}
        <svg 
          className="absolute inset-0 w-full h-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          {/* Gradient fill under curve */}
          <defs>
            <linearGradient id="lossGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="var(--color-replit-accent)" stopOpacity="0.3" />
              <stop offset="100%" stopColor="var(--color-replit-accent)" stopOpacity="0" />
            </linearGradient>
          </defs>
          
          {/* Area fill */}
          {normalizedData.length > 0 && (
            <path
              d={`${pathD} L ${normalizedData[normalizedData.length - 1]?.x ?? 100} 100 L 0 100 Z`}
              fill="url(#lossGradient)"
            />
          )}
          
          {/* Line */}
          <path
            d={pathD}
            fill="none"
            stroke="var(--color-replit-accent)"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
            className="drop-shadow-sm"
          />
          
          {/* Current point indicator */}
          {normalizedData.length > 0 && (
            <circle
              cx={normalizedData[normalizedData.length - 1]?.x ?? 0}
              cy={100 - (normalizedData[normalizedData.length - 1]?.y ?? 0)}
              r="3"
              fill="var(--color-replit-accent)"
              className="animate-pulse"
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
        
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-[10px] text-replit-textMuted -ml-1 transform -translate-x-full pr-1">
          <span>{max?.toFixed(3) ?? '0'}</span>
          <span>{min?.toFixed(3) ?? '0'}</span>
        </div>
      </div>
      
      {/* X-axis label */}
      <div className="text-center text-xs text-replit-textMuted mt-2">
        Training Steps ({data.length} points)
      </div>
    </div>
  );
}

// ============================================================================
// LIVE METRICS CARDS
// ============================================================================

export function LiveMetricsCards() {
  const metricSeries = useTrainingStore((state) => state.metricSeries);
  const { isTraining } = useTrainingProgress();
  
  const metrics = useMemo(() => {
    const result: Array<{ name: string; value: number; split: string }> = [];
    
    Object.entries(metricSeries).forEach(([_key, series]) => {
      if (series.data.length > 0) {
        result.push({
          name: series.name,
          split: series.split,
          value: series.latest,
        });
      }
    });
    
    return result;
  }, [metricSeries]);
  
  if (metrics.length === 0) {
    return null;
  }
  
  return (
    <div className="grid grid-cols-2 gap-3">
      {metrics.slice(0, 4).map((metric) => (
        <div 
          key={`${metric.name}_${metric.split}`}
          className="bg-replit-surface/35 backdrop-blur rounded-lg border border-replit-border/60 p-3"
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-replit-textMuted capitalize">{metric.name}</span>
            <span className="text-[10px] text-replit-textMuted/60">{metric.split}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold text-replit-text">
              {metric.value.toFixed(4)}
            </span>
            {isTraining && (
              <Zap className="w-3 h-3 text-yellow-500 animate-pulse" />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// COMBINED TRAINING DASHBOARD
// ============================================================================

export default function TrainingDashboard() {
  const logs = useTrainingStore((state) => state.logs);
  
  return (
    <div className="h-full overflow-auto bg-replit-bg p-6 space-y-4">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-replit-text">Training Progress</h2>
        <p className="text-sm text-replit-textMuted">Real-time model training metrics</p>
      </div>
      
      {/* Progress bar */}
      <TrainingProgressBar />
      
      {/* Live metrics */}
      <LiveMetricsCards />
      
      {/* Loss curve */}
      <LossCurve />
      
      {/* Training logs */}
      {logs.length > 0 && (
        <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
          <h3 className="text-sm font-semibold text-replit-text mb-3">Training Logs</h3>
          <div className="max-h-32 overflow-y-auto space-y-1 font-mono text-xs">
            {logs.slice(-10).map((log, i) => (
              <div 
                key={i}
                className={`${
                  log.level === 'ERROR' ? 'text-red-400' : 
                  log.level === 'WARN' ? 'text-yellow-400' : 
                  'text-replit-textMuted'
                }`}
              >
                [{log.level}] {log.text}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
