import { Brain, Moon, Sun, Wifi } from 'lucide-react';
import type { ConnectionStatus } from '../../lib/ws';

interface TopBarProps {
  isBuildReady: boolean;
  isPipelineRunning: boolean;
  onRun: () => void;
  onGenerateData: () => void;
  onExportModel: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
  connectionStatus?: ConnectionStatus;
  onPingBackend?: () => void;
  isPinging?: boolean;
  onNavigateHome: () => void;
}

export default function TopBar({
  isBuildReady,
  isPipelineRunning,
  onRun,
  onGenerateData,
  onExportModel,
  isDark,
  onToggleTheme,
  connectionStatus = 'idle',
  onPingBackend,
  isPinging = false,
  onNavigateHome,
}: TopBarProps) {

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
      {/* Left: Brand / Home */}
      <button 
        onClick={onNavigateHome}
        className="flex items-center gap-3 min-w-0 hover:opacity-80 transition-opacity"
        aria-label="Go Home"
      >
        <div className="relative w-7 h-7 rounded-lg flex items-center justify-center shadow-sm bg-gradient-to-br from-sky-500 to-blue-600">
          <Brain className="w-4 h-4 text-white" />
          <div className="absolute inset-0 blur-xl opacity-30 bg-replit-accent" />
        </div>
        <div className="min-w-0 text-left">
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
      </div>
    </header>
  );
}
