import type { MetricsState } from '../../../../lib/metricsReducer';

interface Stage1DataLoadingProps {
  metricsState: MetricsState;
  stage1ScrollRef: React.RefObject<HTMLDivElement | null>;
  isStageRunning: boolean;
  reducedMotion: boolean;
}

/**
 * Stage 1: Data Loading Component
 * Shows the AutoML assistant's thinking process as it loads and analyzes data
 */
export function Stage1DataLoading({
  metricsState,
  stage1ScrollRef,
  isStageRunning,
  reducedMotion,
}: Stage1DataLoadingProps) {
  const thoughts = metricsState.thinkingByStage.DATA_SOURCE ?? [];

  return (
    <div className="h-full flex flex-col">
      {/* Thinking Panel */}
      <div className="flex-1 rounded-xl bg-replit-surface/35 p-6 pb-6 pt-2 min-h-0 flex flex-col">
        <div className="flex items-center gap-3 mb-3">
          <div className="text-lg font-semibold text-replit-text">AutoML Assistant</div>
          {isStageRunning && <ThinkingIndicator reducedMotion={reducedMotion} size="sm" />}
        </div>

        <div
          ref={stage1ScrollRef}
          className="flex-1 overflow-auto rounded-lg bg-replit-surface/40 p-6 relative"
          style={{
            WebkitMaskImage:
              'linear-gradient(to bottom, rgba(0,0,0,0.25) 0%, rgba(0,0,0,0.6) 10%, rgba(0,0,0,1) 32%, rgba(0,0,0,1) 100%)',
            maskImage:
              'linear-gradient(to bottom, rgba(0,0,0,0.25) 0%, rgba(0,0,0,0.6) 10%, rgba(0,0,0,1) 32%, rgba(0,0,0,1) 100%)',
            WebkitMaskRepeat: 'no-repeat',
            maskRepeat: 'no-repeat',
            WebkitMaskSize: '100% 100%',
            maskSize: '100% 100%',
          }}
        >
          {thoughts.length === 0 ? (
            <div className="flex items-center gap-3 text-replit-textMuted">
              <ThinkingIndicator reducedMotion={reducedMotion} size="md" />
              <span className="text-base">Thinking...</span>
            </div>
          ) : (
            <div className="space-y-4">
              {thoughts.slice(-50).map((msg, idx) => (
                <div key={`${idx}-${msg.slice(0, 20)}`} className="text-base leading-relaxed text-replit-text">
                  <span className="text-replit-accent font-medium">▸ </span>
                  {msg}
                </div>
              ))}
              {isStageRunning && (
                <div className="flex items-center gap-2 text-replit-textMuted">
                  <ThinkingIndicator reducedMotion={reducedMotion} size="sm" />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Reusable thinking indicator component
function ThinkingIndicator({
  reducedMotion,
  size = 'md',
}: {
  reducedMotion: boolean;
  size?: 'sm' | 'md';
}) {
  const sizeClass = size === 'sm' ? 'h-2 w-2' : 'h-3 w-3';
  const colors = size === 'sm' 
    ? ['bg-replit-accent/60', 'bg-replit-success/50', 'bg-replit-warning/50']
    : ['bg-replit-accent/50', 'bg-replit-success/40', 'bg-replit-warning/40'];

  return (
    <div className={reducedMotion ? 'grid grid-cols-3 gap-0.5 opacity-80' : 'grid grid-cols-3 gap-0.5 opacity-90'}>
      {Array.from({ length: 9 }).map((_, i) => (
        <div
          key={i}
          className={
            `${sizeClass} rounded-[2px] ${colors[i % colors.length]} ` +
            (reducedMotion ? '' : 'animate-[matrixPulse_900ms_ease-in-out_infinite]')
          }
          style={!reducedMotion ? { animationDelay: `${i * 70}ms` } : undefined}
        />
      ))}
    </div>
  );
}
