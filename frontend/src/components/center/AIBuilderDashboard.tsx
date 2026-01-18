import type { FileSystemNode } from '../../lib/types';
import { Activity } from 'lucide-react';

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

  const metrics = readFile('/artifacts/metrics.json');
  const confusionMatrix = readFile('/artifacts/confusion_matrix.json');

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
        {/* Training Loss History - Main Focus */}
        <div className="col-span-12 lg:col-span-8 bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-6 flex flex-col">
          <h2 className="font-semibold text-replit-text mb-4">Training Loss History</h2>
          <div className="flex-1 w-full flex items-end gap-2 p-4 bg-replit-bg/20 rounded-lg border border-replit-border/60 min-h-[250px]">
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

        {/* Side Panel: Metrics & Confusion Matrix */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
          {/* Metrics Panel */}
          <div className="grid gap-4">
             <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
               <div className="text-replit-textMuted text-sm mb-1">Model Accuracy</div>
               <div className="text-3xl font-bold text-replit-text">{(metrics?.accuracy * 100).toFixed(1)}%</div>
            </div>
             <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
               <div className="text-replit-textMuted text-sm mb-1">F1 Score</div>
               <div className="text-3xl font-bold text-replit-text">{(metrics?.f1_score * 100).toFixed(1)}%</div>
            </div>
          </div>

          {/* Confusion Matrix */}
          <div className="flex-1 bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-6">
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
            ) : (
                <div className="h-40 flex items-center justify-center text-replit-textMuted border border-dashed border-replit-border/60 rounded-lg">
                    No data
                </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
