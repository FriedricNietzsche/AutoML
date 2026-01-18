import { useMetricsStore } from '../../store/metricsStore';
import { Activity, TrendingUp, Target, Zap, BarChart3, PieChart } from 'lucide-react';

export default function MetricsPanel() {
  const {
    taskType,
    classificationMetrics,
    regressionMetrics,
    confusionMatrix,
    featureImportance,
    shapExplanations,
    evaluationComplete,
    isEvaluating,
    hasMetrics,
  } = useMetricsStore();

  if (!hasMetrics()) {
    return (
      <div className="h-full flex items-center justify-center text-replit-textMuted">
        <div className="text-center">
          <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No metrics available yet</p>
          <p className="text-sm mt-2">Metrics will appear after training completes</p>
        </div>
      </div>
    );
  }

  if (isEvaluating) {
    return (
      <div className="h-full flex items-center justify-center text-replit-textMuted">
        <div className="text-center">
          <Activity className="w-12 h-12 mx-auto mb-4 animate-pulse" />
          <p>Computing comprehensive metrics...</p>
        </div>
      </div>
    );
  }

  const renderClassificationMetrics = () => {
    if (!classificationMetrics) return null;

    const mainMetrics = [
      { label: 'Accuracy', value: classificationMetrics.accuracy, icon: Target },
      { label: 'Precision', value: classificationMetrics.precision, icon: Zap },
      { label: 'Recall', value: classificationMetrics.recall, icon: TrendingUp },
      { label: 'F1 Score', value: classificationMetrics.f1, icon: Activity },
    ];

    const advancedMetrics = [
      { label: 'Balanced Accuracy', value: classificationMetrics.balanced_accuracy },
      { label: 'MCC', value: classificationMetrics.mcc, tooltip: 'Matthews Correlation Coefficient' },
      { label: "Cohen's Kappa", value: classificationMetrics.cohen_kappa },
      ...(classificationMetrics.roc_auc ? [{ label: 'ROC AUC', value: classificationMetrics.roc_auc }] : []),
      ...(classificationMetrics.average_precision ? [{ label: 'Avg Precision', value: classificationMetrics.average_precision }] : []),
    ];

    return (
      <>
        {/* Primary Metrics */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {mainMetrics.map(({ label, value, icon: Icon }) => (
            <div key={label} className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-replit-textMuted">{label}</span>
                <Icon className="w-4 h-4 text-replit-accent" />
              </div>
              <div className="text-2xl font-bold text-replit-text">
                {(value * 100).toFixed(1)}%
              </div>
              <div className="mt-2 h-1 bg-replit-bg rounded-full overflow-hidden">
                <div 
                  className="h-full bg-replit-accent transition-all duration-500"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Advanced Metrics */}
        <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4 mb-6">
          <h3 className="text-sm font-semibold text-replit-text mb-3 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Advanced Metrics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {advancedMetrics.map(({ label, value, tooltip }) => (
              <div key={label} className="text-center" title={tooltip}>
                <div className="text-xs text-replit-textMuted mb-1">{label}</div>
                <div className="text-lg font-semibold text-replit-text">
                  {(value * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Class Distribution */}
        {classificationMetrics.class_distribution && (
          <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4 mb-6">
            <h3 className="text-sm font-semibold text-replit-text mb-3 flex items-center gap-2">
              <PieChart className="w-4 h-4" />
              Class Distribution
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(classificationMetrics.class_distribution).map(([cls, count]) => (
                <div key={cls} className="bg-replit-bg/40 rounded-lg p-3">
                  <div className="text-xs text-replit-textMuted">Class {cls}</div>
                  <div className="text-xl font-bold text-replit-text">{count}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Per-class Metrics */}
        {classificationMetrics.precision_per_class && (
          <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4 mb-6">
            <h3 className="text-sm font-semibold text-replit-text mb-3">Per-Class Performance</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-replit-border/40">
                    <th className="text-left py-2 text-replit-textMuted">Class</th>
                    <th className="text-right py-2 text-replit-textMuted">Precision</th>
                    <th className="text-right py-2 text-replit-textMuted">Recall</th>
                    <th className="text-right py-2 text-replit-textMuted">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {classificationMetrics.precision_per_class.map((prec, idx) => (
                    <tr key={idx} className="border-b border-replit-border/20">
                      <td className="py-2 text-replit-text">
                        {classificationMetrics.class_labels?.[idx]?.toString() || `Class ${idx}`}
                      </td>
                      <td className="text-right text-replit-text">{(prec * 100).toFixed(1)}%</td>
                      <td className="text-right text-replit-text">
                        {classificationMetrics.recall_per_class ? (classificationMetrics.recall_per_class[idx] * 100).toFixed(1) : '-'}%
                      </td>
                      <td className="text-right text-replit-text">
                        {classificationMetrics.f1_per_class ? (classificationMetrics.f1_per_class[idx] * 100).toFixed(1) : '-'}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </>
    );
  };

  const renderRegressionMetrics = () => {
    if (!regressionMetrics) return null;

    const mainMetrics = [
      { label: 'R² Score', value: regressionMetrics.r2, format: (v: number) => (v * 100).toFixed(1) + '%' },
      { label: 'RMSE', value: regressionMetrics.rmse, format: (v: number) => v.toFixed(4) },
      { label: 'MAE', value: regressionMetrics.mae, format: (v: number) => v.toFixed(4) },
      { label: 'Median AE', value: regressionMetrics.median_ae, format: (v: number) => v.toFixed(4) },
    ];

    const advancedMetrics = [
      { label: 'MSE', value: regressionMetrics.mse, format: (v: number) => v.toFixed(4) },
      { label: 'Explained Var', value: regressionMetrics.explained_variance, format: (v: number) => (v * 100).toFixed(1) + '%' },
      { label: 'Max Error', value: regressionMetrics.max_error, format: (v: number) => v.toFixed(4) },
      ...(regressionMetrics.mape ? [{ label: 'MAPE', value: regressionMetrics.mape, format: (v: number) => v.toFixed(2) + '%' }] : []),
    ];

    return (
      <>
        {/* Primary Metrics */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {mainMetrics.map(({ label, value, format }) => (
            <div key={label} className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
              <div className="text-sm text-replit-textMuted mb-2">{label}</div>
              <div className="text-2xl font-bold text-replit-text">
                {format(value)}
              </div>
            </div>
          ))}
        </div>

        {/* Advanced Metrics */}
        <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4 mb-6">
          <h3 className="text-sm font-semibold text-replit-text mb-3">Additional Metrics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {advancedMetrics.map(({ label, value, format }) => (
              <div key={label} className="text-center">
                <div className="text-xs text-replit-textMuted mb-1">{label}</div>
                <div className="text-lg font-semibold text-replit-text">
                  {format(value)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </>
    );
  };

  const renderConfusionMatrix = () => {
    if (!confusionMatrix || !confusionMatrix.matrix) return null;

    const matrix = confusionMatrix.matrix;
    const labels = confusionMatrix.labels || matrix.map((_, i) => `Class ${i}`);

    return (
      <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4 mb-6">
        <h3 className="text-sm font-semibold text-replit-text mb-3">Confusion Matrix</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className="p-2"></th>
                {labels.map((label, i) => (
                  <th key={i} className="p-2 text-center text-replit-textMuted">
                    Pred {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => (
                <tr key={i}>
                  <td className="p-2 text-right text-replit-textMuted">
                    True {labels[i]}
                  </td>
                  {row.map((value, j) => {
                    const total = row.reduce((sum, v) => sum + v, 0);
                    const percentage = total > 0 ? (value / total) * 100 : 0;
                    const isCorrect = i === j;
                    
                    return (
                      <td
                        key={j}
                        className={`p-3 text-center font-semibold ${
                          isCorrect
                            ? 'bg-replit-success/20 text-replit-text'
                            : 'bg-replit-bg/40 text-replit-textMuted'
                        }`}
                      >
                        <div>{value}</div>
                        <div className="text-xs opacity-75">
                          {percentage.toFixed(0)}%
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderFeatureImportance = () => {
    if (!featureImportance || !featureImportance.features.length) return null;

    const topFeatures = featureImportance.features.slice(0, 10);
    const maxImportance = Math.max(...topFeatures.map(f => f.importance));

    return (
      <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4 mb-6">
        <h3 className="text-sm font-semibold text-replit-text mb-3">
          Feature Importance (Top 10)
        </h3>
        <div className="space-y-2">
          {topFeatures.map((item, idx) => (
            <div key={idx} className="flex items-center gap-3">
              <div className="text-xs text-replit-textMuted w-4">{idx + 1}</div>
              <div className="flex-1">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-replit-text truncate">{item.feature}</span>
                  <span className="text-xs text-replit-textMuted ml-2">
                    {item.importance.toFixed(4)}
                  </span>
                </div>
                <div className="h-2 bg-replit-bg rounded-full overflow-hidden">
                  <div
                    className="h-full bg-replit-accent transition-all duration-500"
                    style={{ width: `${(item.importance / maxImportance) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="h-full overflow-auto bg-replit-bg p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-replit-text mb-2">Evaluation Metrics</h2>
        <p className="text-sm text-replit-textMuted">
          Comprehensive model performance analysis
          {evaluationComplete && ` • ${evaluationComplete.task_type === 'classification' ? 'Classification' : 'Regression'}`}
        </p>
      </div>

      {/* Metrics based on task type */}
      {taskType === 'classification' && renderClassificationMetrics()}
      {taskType === 'regression' && renderRegressionMetrics()}

      {/* Confusion Matrix */}
      {renderConfusionMatrix()}

      {/* Feature Importance */}
      {renderFeatureImportance()}

      {/* SHAP Explanations - Enhanced Display */}
      {shapExplanations && (
        <div className="bg-replit-surface/35 backdrop-blur rounded-xl border border-replit-border/60 p-4">
          <h3 className="text-sm font-semibold text-replit-text mb-3 flex items-center gap-2">
            <span className="text-yellow-400">✨</span> SHAP Explanations
          </h3>
          
          {shapExplanations.available ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm text-replit-success bg-replit-success/10 p-2 rounded-lg">
                ✅ SHAP analysis available - {shapExplanations.feature_names?.length || 0} features
              </div>
              
              {/* SHAP Importance Ranking */}
              {shapExplanations.importance_ranking && shapExplanations.importance_ranking.length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-replit-textMuted mb-2">SHAP Global Importance</h4>
                  <div className="space-y-2">
                    {shapExplanations.importance_ranking.slice(0, 8).map((item, idx) => {
                      const maxImportance = Math.max(...shapExplanations.importance_ranking!.slice(0, 8).map(f => f.importance));
                      return (
                        <div key={idx} className="flex items-center gap-2">
                          <div className="text-xs text-replit-textMuted w-4">{idx + 1}</div>
                          <div className="flex-1">
                            <div className="flex justify-between items-center mb-0.5">
                              <span className="text-xs text-replit-text truncate max-w-[120px]">{item.feature}</span>
                              <span className="text-xs text-yellow-400">{item.importance.toFixed(3)}</span>
                            </div>
                            <div className="h-1.5 bg-replit-bg rounded-full overflow-hidden">
                              <div
                                className="h-full bg-gradient-to-r from-yellow-500 to-orange-500"
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
              
              {/* Alternative: Global importance dict */}
              {shapExplanations.global_importance && !shapExplanations.importance_ranking && (
                <div>
                  <h4 className="text-xs font-medium text-replit-textMuted mb-2">Global Feature Impact</h4>
                  <div className="space-y-2">
                    {Object.entries(shapExplanations.global_importance)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 8)
                      .map(([feature, importance], idx) => {
                        const maxImportance = Math.max(...Object.values(shapExplanations.global_importance!));
                        return (
                          <div key={idx} className="flex items-center gap-2">
                            <div className="text-xs text-replit-textMuted w-4">{idx + 1}</div>
                            <div className="flex-1">
                              <div className="flex justify-between items-center mb-0.5">
                                <span className="text-xs text-replit-text truncate max-w-[120px]">{feature}</span>
                                <span className="text-xs text-yellow-400">{importance.toFixed(3)}</span>
                              </div>
                              <div className="h-1.5 bg-replit-bg rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-gradient-to-r from-yellow-500 to-orange-500"
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
            <div className="text-sm text-replit-textMuted bg-replit-bg/40 p-2 rounded-lg">
              ❌ {shapExplanations.message || 'SHAP not available - install with: pip install shap'}
            </div>
          )}
        </div>
      )}

      {/* Evaluation Complete Summary */}
      {evaluationComplete && (
        <div className="bg-gradient-to-r from-replit-success/10 to-replit-accent/10 rounded-xl border border-replit-success/40 p-4 mt-4">
          <h3 className="text-sm font-semibold text-replit-text mb-2 flex items-center gap-2">
            ✅ Evaluation Complete
          </h3>
          <div className="text-2xl font-bold text-replit-text mb-1">
            {evaluationComplete.primary_metric}: {(evaluationComplete.primary_value * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-replit-textMuted">
            {evaluationComplete.task_type === 'classification' ? 'Classification' : 'Regression'} • 
            {evaluationComplete.artifacts?.length || 0} artifacts
            {evaluationComplete.shap_available && ' • SHAP ✓'}
          </div>
        </div>
      )}
    </div>
  );
}
