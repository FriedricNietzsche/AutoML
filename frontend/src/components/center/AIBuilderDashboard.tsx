import type { FileSystemNode } from '../../lib/types';
import { Activity, CheckCircle, Settings } from 'lucide-react';
import { Fragment } from 'react';

interface AIBuilderDashboardProps {
  files: FileSystemNode[];
}

export default function AIBuilderDashboard({ files }: AIBuilderDashboardProps) {
  // Helper to read file content from VFS tree
  const readFile = (path: string) => {
    const findNode = (nodes: FileSystemNode[]): FileSystemNode | undefined => {
      for (const node of nodes) {
        if (node.path === path) return node;
        if (node.children) {
          const found = findNode(node.children);
          if (found) return found;
        }
      }
    };
    const node = findNode(files);
    if (!node || !node.content) return null;
    try {
      return JSON.parse(node.content);
    } catch {
      return null;
    }
  };

  const pipeline = readFile('/config/pipeline.json');
  const metrics = readFile('/artifacts/metrics.json');
  const confusionMatrix = readFile('/artifacts/confusion_matrix.json');

  if (!pipeline) return <div className="p-8 text-center text-replit-textMuted">Initializing Dashboard...</div>;

  const maxLoss = metrics?.loss_history ? Math.max(...metrics.loss_history, 0.1) : 1;

  return (
    <div className="h-full flex flex-col bg-replit-bg overflow-auto">
      {/* Header */}
      <div className="bg-replit-surface/40 backdrop-blur border-b border-replit-border/60 px-6 py-4 flex justify-between items-center sticky top-0 z-10">
        <div>
          <h1 className="text-xl font-bold text-replit-text">AI Builder Dashboard</h1>
          <p className="text-sm text-replit-textMuted">Live Training Monitor</p>
        </div>
        <div className="flex gap-2">
          <span className="px-3 py-1 bg-replit-success/15 text-replit-text rounded-full text-xs font-medium flex items-center gap-1 border border-replit-border/60">
            <Activity size={14} /> Agent Active
          </span>
        </div>
      </div>

      <div className="p-6 grid grid-cols-12 gap-6">
        {/* Pipeline Visualization */}
        <div className="col-span-12 lg:col-span-8 bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-6">
          <h2 className="font-semibold text-replit-text mb-6 flex items-center gap-2">
            <Settings size={18} /> Pipeline Status
          </h2>
          <div className="flex items-start w-full">
            {pipeline.nodes.map((node: { id: string; label: string; status: string; progress: number }, index: number) => {
              const isActive = node.status === 'running';
              const isDone = node.status === 'completed';
              const isFailed = node.status === 'failed';
              
              let colorClass = 'bg-replit-surface/40 text-replit-textMuted border-replit-border/60';
              if (isActive) colorClass = 'bg-replit-accent/90 text-white border-replit-accent/90 ring-4 ring-replit-accent/15';
              if (isDone) colorClass = 'bg-replit-success/80 text-white border-replit-success/80';
              if (isFailed) colorClass = 'bg-replit-warning/70 text-white border-replit-warning/70';

              const connectorClass = isDone
                ? 'bg-replit-success/70'
                : isActive
                  ? 'bg-replit-accent/60'
                  : isFailed
                    ? 'bg-replit-warning/70'
                    : 'bg-replit-border/70';

              const hasNext = index < pipeline.nodes.length - 1;

              return (
                <Fragment key={node.id}>
                  <div className="flex flex-col items-center gap-1.5 w-16 min-w-0 shrink-0">
                    <div className={`relative w-7 h-7 rounded-full border-2 flex items-center justify-center transition-all duration-300 z-10 ${colorClass}`}>
                      {isDone ? <CheckCircle size={14} /> : <span className="text-[11px] font-semibold">{index + 1}</span>}
                      {isActive ? (
                        <div
                          aria-hidden
                          className="absolute -inset-1 rounded-full border-2 border-yellow-300/80 border-t-transparent animate-spin"
                        />
                      ) : null}
                    </div>
                    <span className={`text-[10px] leading-tight text-center font-medium ${isActive ? 'text-replit-text' : 'text-replit-textMuted'}`}>
                      {node.label}
                    </span>
                  </div>
                  {hasNext ? <div className={`mt-3 h-0.5 flex-1 mx-2 rounded-full ${connectorClass}`} /> : null}
                </Fragment>
              );
            })}
          </div>
        </div>

        {/* Metrics Panel */}
        <div className="col-span-12 lg:col-span-4 grid gap-4">
           <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
             <div className="text-replit-textMuted text-sm mb-1">Model Accuracy</div>
             <div className="text-3xl font-bold text-replit-text">{(metrics?.accuracy * 100).toFixed(1)}%</div>
          </div>
           <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
             <div className="text-replit-textMuted text-sm mb-1">F1 Score</div>
             <div className="text-3xl font-bold text-replit-text">{(metrics?.f1_score * 100).toFixed(1)}%</div>
          </div>
           <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
             <div className="text-replit-textMuted text-sm mb-1">AUC-ROC</div>
             <div className="text-3xl font-bold text-replit-text">{(metrics?.auc_roc * 100).toFixed(1)}%</div>
          </div>
        </div>

        {/* Training Loss Chart (CSS Only) */}
        <div className="col-span-12 lg:col-span-8 bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-6">
          <h2 className="font-semibold text-replit-text mb-4">Training Loss History</h2>
          <div className="h-64 w-full flex items-end gap-2 p-4 bg-replit-bg/20 rounded-lg border border-replit-border/60">
            {metrics?.loss_history?.length > 0 ? (
               metrics.loss_history.map((val: number, i: number) => (
                  <div key={i} className="flex-1 flex flex-col items-center gap-1 group relative">
                     <div 
                        className="w-full bg-replit-accent/80 rounded-t opacity-80 hover:opacity-100 transition-all"
                        style={{ height: `${(val / maxLoss) * 100}%` }}
                     />
                     <span className="text-xs text-replit-textMuted absolute -bottom-5">E{i+1}</span>
                  </div>
               ))
            ) : (
              <div className="h-full w-full flex items-center justify-center text-replit-textMuted">
                Waiting for training execution...
              </div>
            )}
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className="col-span-12 lg:col-span-4 bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-6">
          <h2 className="font-semibold text-replit-text mb-4">Confusion Matrix</h2>
          {confusionMatrix ? (
            <div className="grid grid-cols-2 gap-2 text-center text-sm">
                <div className="bg-replit-bg/20 p-2 rounded border border-replit-border/60"></div>
                <div className="bg-replit-bg/20 p-2 rounded font-semibold text-replit-text border border-replit-border/60">Pred N</div>
                <div className="bg-replit-bg/20 p-2 rounded font-semibold text-replit-text border border-replit-border/60">Pred P</div>
                
                <div className="bg-replit-bg/20 p-2 rounded font-semibold text-replit-text flex items-center justify-center border border-replit-border/60">Actual N</div>
                <div className="bg-replit-surface/25 p-4 rounded text-replit-text font-bold border border-replit-border/60">{confusionMatrix[0][0]}</div>
                <div className="bg-replit-surface/25 p-4 rounded text-replit-text font-bold border border-replit-border/60">{confusionMatrix[0][1]}</div>

                <div className="bg-replit-bg/20 p-2 rounded font-semibold text-replit-text flex items-center justify-center border border-replit-border/60">Actual P</div>
                <div className="bg-replit-surface/25 p-4 rounded text-replit-text font-bold border border-replit-border/60">{confusionMatrix[1][0]}</div>
                <div className="bg-replit-surface/25 p-4 rounded text-replit-text font-bold border border-replit-border/60">{confusionMatrix[1][1]}</div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
