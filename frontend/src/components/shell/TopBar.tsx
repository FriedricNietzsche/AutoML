import { CheckCircle2, Database, Download, Loader2, Lock, Moon, Play, Sun } from 'lucide-react';

interface TopBarProps {
  isBuildReady: boolean;
  isPipelineRunning: boolean;
  onRun: () => void;
  onGenerateData: () => void;
  onExportModel: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
}

export default function TopBar({
  isBuildReady,
  isPipelineRunning,
  onRun,
  onGenerateData,
  onExportModel,
  isDark,
  onToggleTheme,
}: TopBarProps) {
  const runDisabled = !isBuildReady || isPipelineRunning;

  return (
    <header className="h-12 bg-replit-surface border-b border-replit-border flex items-center justify-between px-4 shrink-0">
      {/* Left: Brand */}
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-7 h-7 bg-replit-accent rounded-lg flex items-center justify-center shadow-sm">
          <span className="text-white font-bold text-sm">AI</span>
        </div>
        <div className="min-w-0">
          <div className="text-replit-text font-semibold text-sm leading-tight truncate">AutoAI Builder</div>
          <div className="text-[11px] text-replit-textMuted leading-tight truncate">Frontend-first training simulator</div>
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-2">
        {/* Theme */}
        <button
          onClick={onToggleTheme}
          className="px-2.5 py-1.5 rounded-lg border border-replit-border bg-replit-surface hover:bg-replit-surfaceHover transition-colors text-xs text-replit-text"
          aria-label="Toggle theme"
        >
          {isDark ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
        </button>

        <button
          onClick={onRun}
          disabled={runDisabled}
          className={
            'px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors flex items-center gap-2 ' +
            (runDisabled
              ? 'bg-replit-surfaceHover text-replit-textMuted border border-replit-border/60 cursor-not-allowed'
              : 'bg-replit-accent hover:bg-replit-accentHover text-white')
          }
          aria-label="Run"
          title={!isBuildReady ? 'Locked until build completes' : isPipelineRunning ? 'Build is running…' : 'Run'}
        >
          <Play className="w-4 h-4" />
          <span className="hidden sm:inline">Run</span>
        </button>

        <div
          className={
            'px-2.5 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-2 ' +
            (isBuildReady
              ? 'border-replit-border bg-replit-surface text-replit-success'
              : 'border-replit-border bg-replit-surface text-replit-textMuted')
          }
          aria-label="Build status"
          title={isBuildReady ? 'READY' : 'BUILDING'}
        >
          {isBuildReady ? <CheckCircle2 className="w-4 h-4" /> : isPipelineRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />}
          <span className="hidden md:inline">{isBuildReady ? 'READY' : 'BUILDING'}</span>
          <span className="md:hidden">{isBuildReady ? 'OK' : '...'}</span>
        </div>

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
