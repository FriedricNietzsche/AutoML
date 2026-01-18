import ModelMetricsVisualizer from '../ModelMetricsVisualizer';
import { mapMetricSeries } from '../utils/loaderHelpers';

import type { MetricsState } from '../../../../lib/metricsReducer';

interface Stage4EvaluationProps {
  metricsState: MetricsState;
  metricKind: 'accuracy' | 'f1' | 'rmse';
}

/**
 * Stage 4: Model Evaluation Component
 * Shows accuracy/RMSE metrics visualization
 */
export function Stage4Evaluation({ 
  metricsState,
  metricKind, 
}: Stage4EvaluationProps) {
  const isRmse = metricKind === 'rmse';
  const data = isRmse ? metricsState.rmseSeries : metricsState.accSeries;

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 min-h-0">
        <ModelMetricsVisualizer
          metricKind={metricKind}
          data={mapMetricSeries(data, metricKind)}
        />
      </div>
      <div className="mt-6 text-xs text-replit-textMuted text-center">
        {isRmse ? 'Plotting RMSE curve…' : 'Plotting accuracy curve…'}
      </div>
    </div>
  );
}
