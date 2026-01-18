import TrainingLossVisualizer from '../TrainingLossVisualizer';

import type { MetricsState } from '../../../../lib/metricsReducer';

interface Stage3TrainingProps {
  metricsState: MetricsState;
}

/**
 * Stage 3: Model Training Component
 * Shows the training loss curve visualization
 */
export function Stage3Training({ metricsState }: Stage3TrainingProps) {
  const lossVisible = metricsState.lossSeries;

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 min-h-0">
        <TrainingLossVisualizer
          data={lossVisible.map((p) => ({ 
            epoch: p.epoch, 
            train_loss: p.train_loss, 
            val_loss: p.val_loss 
          }))}
        />
      </div>
      <div className="mt-6 text-xs text-replit-textMuted text-center">
        Plotting loss curve…
      </div>
    </div>
  );
}
