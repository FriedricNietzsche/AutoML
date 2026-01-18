import React from 'react';
import { Brain, CheckCircle2, Database, Download, Loader2, Moon, Sun, Wifi, ArrowRight } from 'lucide-react';
import { useRouter } from '../../router/router';
import type { ConnectionStatus } from '../../lib/ws';
import type { StageID, StageStatus, WaitingConfirmationPayload } from '../../lib/contract';

type StageState = {
  id: StageID;
  index: number;
  status: StageStatus;
  message?: string;
};

interface TopBarProps {
  isBuildReady: boolean;
  onGenerateData: () => void;
  onExportModel: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
  connectionStatus?: ConnectionStatus;
  onPingBackend?: () => void;
  isPinging?: boolean;
  currentStage?: StageID;
  stages?: Record<StageID, StageState>;
  waitingConfirmation?: WaitingConfirmationPayload | null;
  onConfirm?: () => Promise<void>;
}

export default function TopBar({
  isBuildReady,
  onGenerateData,
  onExportModel,
  isDark,
  onToggleTheme,
  connectionStatus = 'idle',
  onPingBackend,
  isPinging = false,
  currentStage,
  stages,
  waitingConfirmation,
  onConfirm,
}: TopBarProps) {
  const [isConfirming, setIsConfirming] = React.useState(false);
  const { navigate } = useRouter();

  // Check if current stage is waiting for confirmation
  const isWaitingConfirmation = 
    currentStage && 
    stages?.[currentStage]?.status === 'WAITING_CONFIRMATION';

  const handleConfirm = async () => {
    if (!onConfirm || isConfirming) return;
    setIsConfirming(true);
    try {
      await onConfirm();
    } finally {
      setIsConfirming(false);
    }
  };

  const statusStyle =
    connectionStatus === 'open'
      ? 'bg-emerald-100 text-emerald-800 border-emerald-200'
      : connectionStatus === 'connecting'
      ? 'bg-amber-100 text-amber-800 border-amber-200'
      : connectionStatus === 'error'
      ? 'bg-red-100 text-red-800 border-red-200'
      : 'bg-replit-surface text-replit-textMuted border-replit-border';

  return (
    <header className="h-12 bg-replit-surface/65 backdrop-blur-xl border-b border-replit-border/70 flex items-center justify-between px-4 shrink-0">
      {/* Left: Brand */}
      <button 
        onClick={() => navigate('/')}
        className="flex items-center gap-3 min-w-0 hover:opacity-80 transition-opacity cursor-pointer"
        aria-label="Go to home"
      >
        <div className="relative w-7 h-7 rounded-lg flex items-center justify-center shadow-sm bg-gradient-to-br from-sky-500 to-blue-600">
          <Brain className="w-4 h-4 text-white" />
          <div className="absolute inset-0 blur-xl opacity-30 bg-replit-accent" />
        </div>
        <div className="min-w-0">
          <div className="text-replit-text font-semibold text-sm leading-tight truncate">AIAI Workspace</div>
          <div className="text-[11px] text-replit-textMuted leading-tight truncate">Build • Train • Iterate</div>
        </div>
      </button>

      {/* Right: Actions */}
      <div className="flex items-center gap-2">
        <div
          className={`px-2.5 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-2 ${statusStyle}`}
          title="Backend WebSocket status"
        >
          <span
            className={`w-2 h-2 rounded-full ${
              connectionStatus === 'open'
                ? 'bg-emerald-500'
                : connectionStatus === 'connecting'
                ? 'bg-amber-500'
                : connectionStatus === 'error'
                ? 'bg-red-500'
                : 'bg-replit-textMuted'
            }`}
          />
          <span className="hidden sm:inline">WS</span>
          <span className="font-mono text-[11px] uppercase">{connectionStatus}</span>
        </div>

        {onPingBackend && (
          <button
            onClick={onPingBackend}
            disabled={isPinging}
            className={
              'px-2.5 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-2 ' +
              (isPinging
                ? 'border-replit-border bg-replit-surface text-replit-textMuted cursor-wait'
                : 'border-replit-border bg-replit-surface hover:bg-replit-surfaceHover')
            }
            aria-label="Send a ping to the backend"
          >
            <Wifi className="w-4 h-4" />
            <span className="hidden sm:inline">{isPinging ? 'Pinging…' : 'Ping'}</span>
          </button>
        )}

        {/* Theme */}
        <button
          onClick={onToggleTheme}
          className="px-2.5 py-1.5 rounded-lg border border-replit-border bg-replit-surface hover:bg-replit-surfaceHover transition-colors text-xs text-replit-text"
          aria-label="Toggle theme"
        >
          {isDark ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
        </button>

        {/* Confirm Button - always visible, disabled when not waiting for confirmation */}
        {onConfirm && (
          <button
            onClick={handleConfirm}
            disabled={!isWaitingConfirmation || isConfirming}
            className={
              'px-4 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-2 ' +
              (!isWaitingConfirmation
                ? 'bg-gray-500/20 text-gray-400 border border-gray-500/30 cursor-not-allowed'
                : isConfirming
                ? 'bg-green-500/70 text-white cursor-wait shadow-lg'
                : 'bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white shadow-lg hover:shadow-xl')
            }
            aria-label="Confirm and continue"
            title={!isWaitingConfirmation ? 'Waiting for stage to complete...' : currentStage ? `Confirm ${currentStage} and continue` : 'Confirm and continue'}
          >
            {isConfirming ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Confirming...</span>
              </>
            ) : (
              <>
                <CheckCircle2 className="w-4 h-4" />
                <span>Confirm</span>
                <ArrowRight className="w-3 h-3" />
              </>
            )}
          </button>
        )}

        {isBuildReady && (
          <>
            <button
              onClick={onGenerateData}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-2 border border-replit-border bg-replit-surface hover:bg-replit-surfaceHover"
              aria-label="Generate Data"
            >
              <Database className="w-4 h-4" />
              <span className="hidden sm:inline">Generate Data</span>
            </button>

            <button
              onClick={onExportModel}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-2 border border-replit-border bg-replit-surface hover:bg-replit-surfaceHover"
              aria-label="Export Model"
            >
              <Download className="w-4 h-4" />
              <span className="hidden sm:inline">Export</span>
            </button>
          </>
        )}
      </div>
    </header>
  );
}
