import type { FileSystemNode } from '../../lib/types';
import { Activity, CheckCircle, Settings, BarChart3, Target, TrendingUp, Zap, PieChart, Sparkles, Info } from 'lucide-react';
import { Fragment } from 'react';
import { useMetricsStore } from '../../store/metricsStore';

interface AIBuilderDashboardProps {
  files: FileSystemNode[];
}

export default function AIBuilderDashboard({ files }: AIBuilderDashboardProps) {
  // Get real-time metrics from store
  const {
    taskType,
    classificationMetrics,
    regressionMetrics,
    confusionMatrix: storeConfusionMatrix,
    featureImportance,
    shapExplanations,
    evaluationComplete,
    metricHistory,
    isEvaluating,
    hasMetrics,
  } = useMetricsStore();

  // Helper to read file content from VFS tree (for pipeline status)
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
  
  // Use real-time metrics from store, fall back to file-based metrics
  const fileMetrics = readFile('/artifacts/metrics.json');
  const fileConfusionMatrix = readFile('/artifacts/confusion_matrix.json');
  
  // Derive metrics from store or files
  const metrics = classificationMetrics || regressionMetrics ? {
    accuracy: classificationMetrics?.accuracy ?? fileMetrics?.accuracy ?? 0,
    f1_score: classificationMetrics?.f1 ?? fileMetrics?.f1_score ?? 0,
    auc_roc: classificationMetrics?.roc_auc ?? fileMetrics?.auc_roc ?? 0,
    precision: classificationMetrics?.precision ?? 0,
    recall: classificationMetrics?.recall ?? 0,
    mcc: classificationMetrics?.mcc ?? 0,
    r2: regressionMetrics?.r2 ?? 0,
    rmse: regressionMetrics?.rmse ?? 0,
    mae: regressionMetrics?.mae ?? 0,
    loss_history: Object.values(metricHistory)
      .filter(h => h.name === 'loss')
      .flatMap(h => h.values.map(v => v.value)),
  } : fileMetrics;
  
  const confusionMatrix = storeConfusionMatrix?.matrix || fileConfusionMatrix;

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

        {/* Metrics Panel - Dynamic based on task type */}
        <div className="col-span-12 lg:col-span-4 grid gap-4">
          {/* Show evaluating status */}
          {isEvaluating && (
            <div className="bg-replit-accent/10 backdrop-blur rounded-xl border border-replit-accent/40 p-4">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-replit-accent animate-pulse" />
                <span className="text-sm text-replit-text">Computing metrics...</span>
              </div>
            </div>
          )}
          
          {/* Classification Metrics */}
          {(taskType === 'classification' || (!taskType && metrics?.accuracy !== undefined)) && (
            <>
              <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-replit-textMuted text-sm">Accuracy</div>
                  <Target className="w-4 h-4 text-replit-accent" />
                </div>
                <div className="text-3xl font-bold text-replit-text">{((metrics?.accuracy ?? 0) * 100).toFixed(1)}%</div>
                <div className="mt-2 h-1 bg-replit-bg rounded-full overflow-hidden">
                  <div className="h-full bg-replit-accent transition-all duration-500" style={{ width: `${(metrics?.accuracy ?? 0) * 100}%` }} />
                </div>
              </div>
              <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-replit-textMuted text-sm">F1 Score</div>
                  <Zap className="w-4 h-4 text-replit-accent" />
                </div>
                <div className="text-3xl font-bold text-replit-text">{((metrics?.f1_score ?? 0) * 100).toFixed(1)}%</div>
              </div>
              <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-replit-textMuted text-sm">AUC-ROC</div>
                  <TrendingUp className="w-4 h-4 text-replit-accent" />
                </div>
                <div className="text-3xl font-bold text-replit-text">{((metrics?.auc_roc ?? 0) * 100).toFixed(1)}%</div>
              </div>
            </>
          )}
          
          {/* Regression Metrics */}
          {taskType === 'regression' && (
            <>
              <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-replit-textMuted text-sm">R² Score</div>
                  <Target className="w-4 h-4 text-replit-accent" />
                </div>
                <div className="text-3xl font-bold text-replit-text">{((metrics?.r2 ?? 0) * 100).toFixed(1)}%</div>
                <div className="mt-2 h-1 bg-replit-bg rounded-full overflow-hidden">
                  <div className="h-full bg-replit-accent transition-all duration-500" style={{ width: `${Math.max(0, (metrics?.r2 ?? 0) * 100)}%` }} />
                </div>
              </div>
              <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-replit-textMuted text-sm">RMSE</div>
                  <BarChart3 className="w-4 h-4 text-replit-accent" />
                </div>
                <div className="text-3xl font-bold text-replit-text">{(metrics?.rmse ?? 0).toFixed(4)}</div>
              </div>
              <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="text-replit-textMuted text-sm">MAE</div>
                  <TrendingUp className="w-4 h-4 text-replit-accent" />
                </div>
                <div className="text-3xl font-bold text-replit-text">{(metrics?.mae ?? 0).toFixed(4)}</div>
              </div>
            </>
          )}
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

        {/* Confusion Matrix - Dynamic for any class count */}
        <div className="col-span-12 lg:col-span-4 bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-6">
          <h2 className="font-semibold text-replit-text mb-4 flex items-center gap-2">
            <PieChart size={16} /> Confusion Matrix
          </h2>
          {confusionMatrix && confusionMatrix.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-center text-sm">
                <thead>
                  <tr>
                    <th className="p-2 text-replit-textMuted"></th>
                    {confusionMatrix[0]?.map((_: number, i: number) => (
                      <th key={i} className="p-2 text-replit-textMuted text-xs">
                        {storeConfusionMatrix?.labels?.[i] ?? `Pred ${i}`}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {confusionMatrix.map((row: number[], i: number) => (
                    <tr key={i}>
                      <td className="p-2 text-replit-textMuted text-xs font-semibold">
                        {storeConfusionMatrix?.labels?.[i] ?? `True ${i}`}
                      </td>
                      {row.map((val: number, j: number) => {
                        const total = row.reduce((sum: number, v: number) => sum + v, 0);
                        const pct = total > 0 ? (val / total) * 100 : 0;
                        const isCorrect = i === j;
                        return (
                          <td
                            key={j}
                            className={`p-3 font-bold ${
                              isCorrect
                                ? 'bg-replit-success/20 text-replit-text'
                                : val > 0
                                  ? 'bg-replit-warning/10 text-replit-textMuted'
                                  : 'bg-replit-bg/20 text-replit-textMuted'
                            }`}
                          >
                            <div>{val}</div>
                            <div className="text-xs opacity-60">{pct.toFixed(0)}%</div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center text-replit-textMuted py-8">
              <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No confusion matrix available</p>
            </div>
          )}
        </div>

        {/* Feature Importance */}
        {featureImportance && featureImportance.features.length > 0 && (
          <div className="col-span-12 bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-6">
            <h2 className="font-semibold text-replit-text mb-4 flex items-center gap-2">
              <BarChart3 size={16} /> Feature Importance (Top 10)
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {featureImportance.features.slice(0, 10).map((item, idx) => {
                const maxImportance = Math.max(...featureImportance.features.slice(0, 10).map(f => f.importance));
                return (
                  <div key={idx} className="flex items-center gap-3">
                    <div className="text-xs text-replit-textMuted w-5 text-right">{idx + 1}</div>
                    <div className="flex-1">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm text-replit-text truncate max-w-[200px]">{item.feature}</span>
                        <span className="text-xs text-replit-textMuted ml-2">{item.importance.toFixed(4)}</span>
                      </div>
                      <div className="h-2 bg-replit-bg rounded-full overflow-hidden">
                        <div
                          className="h-full bg-replit-accent transition-all duration-500"
                          style={{ width: `${(item.importance / maxImportance) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* SHAP Explanations Section */}
        {shapExplanations && (
          <div className="col-span-12 bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-6">
            <h2 className="font-semibold text-replit-text mb-4 flex items-center gap-2">
              <Sparkles size={16} className="text-yellow-400" /> SHAP Explanations
            </h2>
            {shapExplanations.available ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-sm text-replit-success bg-replit-success/10 p-3 rounded-lg">
                  <CheckCircle size={16} />
                  <span>SHAP analysis completed - {shapExplanations.feature_names?.length || 0} features analyzed</span>
                </div>
                
                {/* SHAP Global Importance */}
                {shapExplanations.importance_ranking && shapExplanations.importance_ranking.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium text-replit-text mb-3 flex items-center gap-2">
                      <Info size={14} />
                      SHAP Global Feature Importance (Model-Agnostic)
                    </h3>
                    <p className="text-xs text-replit-textMuted mb-3">
                      SHAP values show the average impact of each feature on model predictions, accounting for feature interactions.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {shapExplanations.importance_ranking.slice(0, 10).map((item, idx) => {
                        const maxImportance = Math.max(...shapExplanations.importance_ranking!.slice(0, 10).map(f => f.importance));
                        return (
                          <div key={idx} className="flex items-center gap-3">
                            <div className="text-xs text-replit-textMuted w-5 text-right">{idx + 1}</div>
                            <div className="flex-1">
                              <div className="flex justify-between items-center mb-1">
                                <span className="text-sm text-replit-text truncate max-w-[200px]">{item.feature}</span>
                                <span className="text-xs text-yellow-400 ml-2">{item.importance.toFixed(4)}</span>
                              </div>
                              <div className="h-2 bg-replit-bg rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-gradient-to-r from-yellow-500 to-orange-500 transition-all duration-500"
                                  style={{ width: `${(item.importance / maxImportance) * 100}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                
                {/* Global importance as alternative */}
                {shapExplanations.global_importance && !shapExplanations.importance_ranking && (
                  <div>
                    <h3 className="text-sm font-medium text-replit-text mb-3">Global Feature Importance</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {Object.entries(shapExplanations.global_importance)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10)
                        .map(([feature, importance], idx) => {
                          const maxImportance = Math.max(...Object.values(shapExplanations.global_importance!));
                          return (
                            <div key={idx} className="flex items-center gap-3">
                              <div className="text-xs text-replit-textMuted w-5 text-right">{idx + 1}</div>
                              <div className="flex-1">
                                <div className="flex justify-between items-center mb-1">
                                  <span className="text-sm text-replit-text truncate max-w-[200px]">{feature}</span>
                                  <span className="text-xs text-yellow-400 ml-2">{importance.toFixed(4)}</span>
                                </div>
                                <div className="h-2 bg-replit-bg rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-yellow-500 to-orange-500 transition-all duration-500"
                                    style={{ width: `${(importance / maxImportance) * 100}%` }}
                                  />
                                </div>
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-replit-textMuted bg-replit-bg/40 p-3 rounded-lg">
                <Info size={16} />
                <span>{shapExplanations.message || 'SHAP library not installed - install with: pip install shap'}</span>
              </div>
            )}
          </div>
        )}

        {/* Evaluation Summary */}
        {evaluationComplete && hasMetrics() && (
          <div className="col-span-12 bg-gradient-to-r from-replit-success/10 to-replit-accent/10 backdrop-blur rounded-xl border border-replit-success/40 p-6">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="font-semibold text-replit-text mb-2 flex items-center gap-2">
                  <CheckCircle size={18} className="text-replit-success" /> Training Complete
                </h2>
                <p className="text-sm text-replit-textMuted">
                  {evaluationComplete.task_type === 'classification' ? 'Classification' : 'Regression'} model trained successfully
                </p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-replit-text">
                  {evaluationComplete.primary_metric}: {(evaluationComplete.primary_value * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-replit-textMuted mt-1">
                  {evaluationComplete.artifacts?.length || 0} artifacts generated
                  {evaluationComplete.shap_available && ' • SHAP available'}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
