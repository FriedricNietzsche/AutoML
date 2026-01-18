import { useMemo, useState } from 'react';
import { ChevronLeft, ExternalLink, X, Plus } from 'lucide-react';
import { isValidHuggingFaceDatasetLink, type BuildSession } from '../../lib/buildSession';
import StageTimeline from './StageTimeline';

interface AIBuilderPanelProps {
  session: BuildSession;
  onCollapse: () => void;
  onEditSession: () => void;
  onUpdateSession: (patch: Partial<BuildSession>) => void;
}

export default function AIBuilderPanel({
  session,
  onCollapse,
  onEditSession,
  onUpdateSession,
}: AIBuilderPanelProps) {
  const [datasetDraft, setDatasetDraft] = useState('');
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [isAddingDataset, setIsAddingDataset] = useState(false);

  const goalSummary = useMemo(() => {
    const g = session.goalPrompt.trim();
    if (!g) return '—';
    // Show full goal now since we have space
    return g;
  }, [session.goalPrompt]);

  const datasetLinks = useMemo(() => {
    const fromArray = (session.datasetLinks ?? []).map((s) => s.trim()).filter(Boolean);
    if (fromArray.length > 0) return fromArray;
    return session.kaggleLink ? [session.kaggleLink] : [];
  }, [session.datasetLinks, session.kaggleLink]);

  const addDataset = () => {
    const next = datasetDraft.trim();
    if (!next) return;
    if (!isValidHuggingFaceDatasetLink(next)) {
      setDatasetError('Invalid dataset link. Use a Hugging Face dataset URL.');
      return;
    }
    if (datasetLinks.includes(next)) {
      setDatasetDraft('');
      setDatasetError(null);
      return;
    }

    const merged = [...datasetLinks, next];
    onUpdateSession({ datasetLinks: merged, kaggleLink: merged[0] });
    setDatasetDraft('');
    setDatasetError(null);
    setIsAddingDataset(false);
  };

  const removeDataset = (idx: number) => {
    const merged = datasetLinks.filter((_, i) => i !== idx);
    onUpdateSession({ datasetLinks: merged.length ? merged : undefined, kaggleLink: merged[0] });
  };

  return (
    <div className="h-full flex flex-col bg-replit-surface border-r border-replit-border">
      {/* Header */}
      <div className="p-3 border-b border-replit-border/60 flex items-center justify-between">
        <div className="min-w-0">
          <div className="text-xs font-bold text-replit-textMuted uppercase tracking-wider">Project</div>
          <div className="text-sm font-semibold text-replit-text truncate">Model Info</div>
        </div>
        <button
          onClick={onCollapse}
          className="p-2 rounded-lg hover:bg-replit-surfaceHover transition-colors text-replit-textMuted"
          aria-label="Collapse input panel"
          title="Collapse (Ctrl+B)"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-6">
        <StageTimeline />

        {/* Model Details */}
        <div className="rounded-xl border border-replit-border bg-replit-bg p-4">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div className="min-w-0">
              <div className="text-xs text-replit-textMuted uppercase tracking-wider mb-1">Model Architecture</div>
              <div className="text-base font-semibold truncate">{session.modelName}</div>
            </div>
            <button
              onClick={onEditSession}
              className="text-xs text-replit-accent hover:text-replit-accentHover inline-flex items-center gap-1 font-medium"
            >
              Edit <ExternalLink className="w-3 h-3" />
            </button>
          </div>

          <div className="mb-4">
            <div className="text-xs text-replit-textMuted uppercase tracking-wider mb-2">Goal</div>
            <div className="text-sm text-replit-text leading-relaxed bg-replit-surface/50 p-3 rounded-lg border border-replit-border/50">
              {goalSummary}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs text-replit-textMuted uppercase tracking-wider">Datasets</div>
              {!isAddingDataset && (
                <button 
                  onClick={() => setIsAddingDataset(true)}
                  className="text-xs flex items-center gap-1 text-replit-accent hover:text-replit-accentHover"
                >
                  <Plus className="w-3 h-3" /> Add
                </button>
              )}
            </div>
            
            {isAddingDataset && (
              <div className="mb-3 animate-in fade-in slide-in-from-top-2 duration-200">
                <div className="flex gap-2">
                  <input
                    value={datasetDraft}
                    onChange={(e) => {
                      setDatasetDraft(e.target.value);
                      if (datasetError) setDatasetError(null);
                    }}
                    placeholder="HuggingFace URL..."
                    className="flex-1 rounded-lg border border-replit-border bg-replit-surface px-3 py-1.5 text-sm outline-none focus:border-replit-accent/80 transition"
                    autoFocus
                    onKeyDown={(e) => e.key === 'Enter' && addDataset()}
                  />
                  <button
                    onClick={addDataset}
                    disabled={!datasetDraft.trim()}
                    className="px-3 py-1.5 rounded-lg bg-replit-accent text-white text-xs font-semibold hover:bg-replit-accentHover disabled:opacity-50"
                  >
                    Add
                  </button>
                  <button
                    onClick={() => {
                      setIsAddingDataset(false);
                      setDatasetDraft('');
                      setDatasetError(null);
                    }}
                    className="p-1.5 rounded-lg hover:bg-replit-surfaceHover text-replit-textMuted"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                {datasetError && <div className="mt-1 text-xs text-replit-warning">{datasetError}</div>}
              </div>
            )}

            {datasetLinks.length > 0 ? (
              <div className="space-y-2">
                {datasetLinks.map((link, idx) => (
                  <div
                    key={`${link}-${idx}`}
                    className="flex items-start gap-2 rounded-lg border border-replit-border bg-replit-surface px-3 py-2 group"
                  >
                    <div className="min-w-0 flex-1 text-xs text-replit-textMuted break-words font-mono">
                      {link.replace('https://huggingface.co/datasets/', '')}
                    </div>
                    <button
                      type="button"
                      onClick={() => removeDataset(idx)}
                      className="shrink-0 p-1 rounded hover:bg-replit-surfaceHover/40 text-replit-textMuted opacity-0 group-hover:opacity-100 transition-opacity"
                      aria-label="Remove dataset"
                      title="Remove"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-replit-textMuted italic bg-replit-surface/30 p-3 rounded-lg border border-replit-border/30 border-dashed text-center">
                No datasets linked.
                <br />
                <span className="text-xs opacity-75">Auto-discovery enabled</span>
              </div>
            )}
          </div>
        </div>
        
        {/* Info Box */}
        <div className="rounded-xl border border-replit-accent/20 bg-replit-accent/5 p-4">
          <h3 className="text-sm font-semibold text-replit-accent mb-2">Pipeline Status</h3>
          <p className="text-xs text-replit-textMuted leading-relaxed">
            The automated pipeline is managing the workflow. You can monitor progress in the Preview panel.
            Model checkpoints are automatically saved to the artifacts directory.
          </p>
        </div>
      </div>
    </div>
  );
}
