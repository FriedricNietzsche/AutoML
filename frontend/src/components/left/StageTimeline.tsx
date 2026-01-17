import { AlertTriangle, ArrowRight, CheckCircle2, CircleDot, PauseCircle, RefreshCw } from 'lucide-react';
import { STAGE_ORDER, type StageStatus } from '../../lib/contract';
import { useProjectStore } from '../../store/projectStore';

const statusColor = (status: StageStatus) => {
  switch (status) {
    case 'COMPLETED':
      return 'text-emerald-500 bg-emerald-500/10 border-emerald-500/30';
    case 'IN_PROGRESS':
      return 'text-sky-500 bg-sky-500/10 border-sky-500/30';
    case 'WAITING_CONFIRMATION':
      return 'text-amber-500 bg-amber-500/10 border-amber-500/30';
    case 'FAILED':
      return 'text-red-500 bg-red-500/10 border-red-500/30';
    case 'SKIPPED':
      return 'text-replit-textMuted bg-replit-surfaceHover border-replit-border';
    default:
      return 'text-replit-textMuted bg-replit-surface border-replit-border';
  }
};

const statusIcon = (status: StageStatus) => {
  switch (status) {
    case 'COMPLETED':
      return <CheckCircle2 className="w-4 h-4" />;
    case 'IN_PROGRESS':
      return <ArrowRight className="w-4 h-4" />;
    case 'WAITING_CONFIRMATION':
      return <PauseCircle className="w-4 h-4" />;
    case 'FAILED':
      return <AlertTriangle className="w-4 h-4" />;
    default:
      return <CircleDot className="w-4 h-4" />;
  }
};

export default function StageTimeline() {
  const {
    stages,
    currentStageId,
    waitingConfirmation,
    connectionStatus,
    confirm,
    hydrate,
    error,
  } = useProjectStore((state) => ({
    stages: state.stages,
    currentStageId: state.currentStageId,
    waitingConfirmation: state.waitingConfirmation,
    connectionStatus: state.connectionStatus,
    confirm: state.confirm,
    hydrate: state.hydrate,
    error: state.error,
  }));

  const confirmDisabled =
    connectionStatus !== 'open' || stages[currentStageId]?.status === 'COMPLETED';

  return (
    <div className="p-3 border-b border-replit-border/60 bg-replit-surface">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-xs font-bold text-replit-textMuted uppercase tracking-wider">Flow</div>
          <div className="text-sm font-semibold text-replit-text">Stage Timeline</div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={hydrate}
            className="px-2.5 py-1.5 rounded-lg border border-replit-border bg-replit-surface hover:bg-replit-surfaceHover text-xs text-replit-text"
            title="Refresh state from backend"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={confirm}
            disabled={confirmDisabled}
            className={
              'px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-2 ' +
              (confirmDisabled
                ? 'bg-replit-surfaceHover text-replit-textMuted border border-replit-border/60 cursor-not-allowed'
                : 'bg-replit-accent hover:bg-replit-accentHover text-white')
            }
            title="Advance to next stage"
          >
            Confirm
          </button>
        </div>
      </div>

      <div className="mt-3 space-y-1.5">
        {STAGE_ORDER.map((stage) => {
          const state = stages[stage.id];
          const status = state?.status ?? 'PENDING';
          return (
            <div
              key={stage.id}
              className="flex items-center justify-between rounded-lg border border-replit-border/60 bg-replit-bg px-3 py-2"
            >
              <div className="flex items-center gap-3 min-w-0">
                <div
                  className={`w-8 h-8 rounded-full border flex items-center justify-center ${statusColor(status)}`}
                  aria-label={status}
                >
                  {statusIcon(status)}
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-replit-text truncate">
                    {stage.label}
                    {stage.id === currentStageId && status === 'IN_PROGRESS' && (
                      <span className="ml-2 text-[11px] text-replit-textMuted">active</span>
                    )}
                  </div>
                  <div className="text-xs text-replit-textMuted truncate">{stage.description}</div>
                </div>
              </div>
              <div className="text-[11px] uppercase font-semibold text-replit-textMuted">{status}</div>
            </div>
          );
        })}
      </div>

      {waitingConfirmation && (
        <div className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-amber-800">
          <div className="text-xs font-semibold uppercase tracking-wide">Waiting confirmation</div>
          <div className="text-sm font-medium mt-1">{waitingConfirmation.summary}</div>
          {waitingConfirmation.next_actions?.length ? (
            <ul className="list-disc list-inside text-xs text-amber-900 mt-1 space-y-0.5">
              {waitingConfirmation.next_actions.map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ul>
          ) : null}
        </div>
      )}

      {error && (
        <div className="mt-2 text-xs text-red-500">Backend sync error: {error}</div>
      )}
    </div>
  );
}
