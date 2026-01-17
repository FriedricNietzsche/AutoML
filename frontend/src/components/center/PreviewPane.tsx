import { useEffect, useState } from 'react';
import { ArrowLeft, ArrowRight, RotateCw, ExternalLink } from 'lucide-react';
import type { FileSystemNode } from '../../lib/types';
import EmptyPreview from './preview/EmptyPreview';
import TrainingLoaderV2 from './preview/TrainingLoaderV2';
import APIDocsPane from './preview/APIDocsPane';
import type { BuildStatus } from '../../lib/buildSession';

interface PreviewPaneProps {
  files: FileSystemNode[];
  isRunning: boolean;
  onSimulationComplete: () => void;
  updateFileContent: (path: string, content: string | ((prev: string) => string)) => void;
  hasSession: boolean;
  sessionStatus: BuildStatus;
}

export default function PreviewPane({ 
  files, 
  isRunning, 
  onSimulationComplete,
  updateFileContent,
  hasSession,
  sessionStatus,
}: PreviewPaneProps) {
  const [completed, setCompleted] = useState(false);

  // Reset completion flag when a new build starts.
  useEffect(() => {
    const shouldReset = !hasSession || isRunning || sessionStatus === 'building';
    if (!shouldReset) return;
    const t = window.setTimeout(() => setCompleted(false), 0);
    return () => window.clearTimeout(t);
  }, [hasSession, isRunning, sessionStatus]);

  // Determine View State
  let viewState: 'empty' | 'processing' | 'docs' = 'empty';
  if (!hasSession) {
    viewState = 'empty';
  } else if (completed) {
    viewState = 'docs';
  } else if (isRunning || sessionStatus === 'building') {
    viewState = 'processing';
  } else {
    viewState = 'docs';
  }

  const handleComplete = () => {
      setCompleted(true);
      onSimulationComplete();
  };

  return (
    <div className="h-full flex flex-col bg-replit-bg/30 backdrop-blur-xl">
      {/* Browser-like Controls */}
      <div className="h-10 bg-replit-surface/60 backdrop-blur border-b border-replit-border/70 flex items-center px-3 gap-2 shrink-0">
        <div className="flex items-center gap-1">
          <button className="p-1.5 hover:bg-replit-surfaceHover rounded transition-colors" disabled>
            <ArrowLeft className="w-4 h-4 text-replit-textMuted" />
          </button>
          <button className="p-1.5 hover:bg-replit-surfaceHover rounded transition-colors" disabled>
            <ArrowRight className="w-4 h-4 text-replit-textMuted" />
          </button>
          <button 
            className="p-1.5 hover:bg-replit-surfaceHover rounded transition-colors" 
            onClick={() => window.location.reload()}
          >
            <RotateCw className="w-4 h-4 text-replit-text" />
          </button>
        </div>

        <div className="flex-1 flex items-center gap-2 px-3 py-1.5 bg-replit-bg/40 backdrop-blur rounded border border-replit-border/70 mx-2">
          <span className="text-xs text-replit-textMuted font-mono truncate flex-1">
             {viewState === 'processing' ? 'https://cluster.ai-builder.dev/jobs/run_8392' : 
              viewState === 'docs' ? 'https://api.ai-builder.dev/v1/docs' : 
              'about:blank'}
          </span>
          <ExternalLink className="w-3 h-3 text-replit-textMuted" />
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden relative">
         {viewState === 'empty' && <EmptyPreview />}
         {viewState === 'processing' && (
          <TrainingLoaderV2 
                onComplete={handleComplete} 
                updateFileContent={updateFileContent} 
            />
         )}
         {viewState === 'docs' && <APIDocsPane files={files} />}
      </div>
    </div>
  );

}
