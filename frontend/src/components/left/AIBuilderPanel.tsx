import { useEffect, useMemo, useRef, useState } from 'react';
import { useReducedMotion } from 'framer-motion';
import { ChevronLeft, ChevronDown, ChevronUp, ExternalLink, Plus, SlidersHorizontal, X, ArrowUp } from 'lucide-react';
import { isValidKaggleDatasetLink, type BuildSession } from '../../lib/buildSession';

interface AIBuilderPanelProps {
  session: BuildSession;
  onCollapse: () => void;
  onEditSession: () => void;
  onUpdateSession: (patch: Partial<BuildSession>) => void;
  onSendMessage: (text: string) => void;
}


function MatrixThinkingLoader({ active }: { active: boolean }) {
  const reduceMotion = useReducedMotion();
  if (!active) return null;

  const colors = ['bg-replit-accent/50', 'bg-replit-success/40', 'bg-replit-warning/40'];

  return (
    <div className="flex items-center gap-3">
      <div className={reduceMotion ? 'grid grid-cols-3 gap-1 opacity-80' : 'grid grid-cols-3 gap-1 opacity-90'} aria-hidden>
        {Array.from({ length: 9 }).map((_, i) => (
          <div
            key={i}
            className={
              `h-2.5 w-2.5 rounded-[3px] ${colors[i % colors.length]} ` +
              (reduceMotion ? '' : 'animate-[matrixPulse_900ms_ease-in-out_infinite]')
            }
            style={!reduceMotion ? { animationDelay: `${i * 70}ms` } : undefined}
          />
        ))}
      </div>
      <div className="text-xs text-replit-textMuted">Thinking…</div>
    </div>
  );
}

export default function AIBuilderPanel({
  session,
  onCollapse,
  onEditSession,
  onUpdateSession,
  onSendMessage,
}: AIBuilderPanelProps) {
  const [timeBudget, setTimeBudget] = useState<'fast' | 'balanced' | 'thorough'>('balanced');
  const [candidates, setCandidates] = useState(6);
  const [datasetDraft, setDatasetDraft] = useState('');
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [messageDraft, setMessageDraft] = useState('');

  const [modelInfoOpen, setModelInfoOpen] = useState(false);
  const [isOptionsOpen, setIsOptionsOpen] = useState(false);
  const [activeOption, setActiveOption] = useState<'none' | 'datasets' | 'constraints'>('none');

  const optionsRef = useRef<HTMLDivElement | null>(null);
  const messageRef = useRef<HTMLTextAreaElement | null>(null);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);

  const goalSummary = useMemo(() => {
    const g = session.goalPrompt.trim();
    if (!g) return '—';
    return g.length > 160 ? g.slice(0, 160) + '…' : g;
  }, [session.goalPrompt]);

  const datasetLinks = useMemo(() => {
    const fromArray = (session.datasetLinks ?? []).map((s) => s.trim()).filter(Boolean);
    if (fromArray.length > 0) return fromArray;
    return session.kaggleLink ? [session.kaggleLink] : [];
  }, [session.datasetLinks, session.kaggleLink]);

  const chatHistory = useMemo(() => session.chatHistory ?? [], [session.chatHistory]);
  const aiThinking = !!session.aiThinking;

  const addDataset = () => {
    const next = datasetDraft.trim();
    if (!next) return;
    if (!isValidKaggleDatasetLink(next)) {
      setDatasetError('Invalid dataset link. Use a Kaggle dataset URL.');
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
  };

  const removeDataset = (idx: number) => {
    const merged = datasetLinks.filter((_, i) => i !== idx);
    onUpdateSession({ datasetLinks: merged.length ? merged : undefined, kaggleLink: merged[0] });
  };

  const sendMessage = () => {
    const text = messageDraft.trim();
    if (!text) return;
    onSendMessage(text);
    setMessageDraft('');
  };

  const autoGrowTextarea = () => {
    const el = messageRef.current;
    if (!el) return;
    const maxPx = 160;
    el.style.height = '0px';
    const nextHeight = Math.min(el.scrollHeight, maxPx);
    el.style.height = `${nextHeight}px`;
    el.style.overflowY = el.scrollHeight > maxPx ? 'auto' : 'hidden';
  };

  useEffect(() => {
    if (!isOptionsOpen) return;

    const onPointerDown = (e: PointerEvent) => {
      const el = optionsRef.current;
      if (!el) return;
      if (e.target instanceof Node && el.contains(e.target)) return;
      setIsOptionsOpen(false);
      setActiveOption('none');
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsOptionsOpen(false);
        setActiveOption('none');
      }
    };

    window.addEventListener('pointerdown', onPointerDown, { capture: true });
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('pointerdown', onPointerDown, true);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [isOptionsOpen]);

  useEffect(() => {
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight });
  }, [chatHistory.length, aiThinking]);

  useEffect(() => {
    autoGrowTextarea();
  }, [messageDraft]);

  return (
    <div className="h-full flex flex-col bg-replit-surface border-r border-replit-border">
      {/* Header */}
      <div className="p-3 border-b border-replit-border/60 flex items-center justify-between">
        <div className="min-w-0">
          <div className="text-xs font-bold text-replit-textMuted uppercase tracking-wider">Input</div>
          <div className="text-sm font-semibold text-replit-text truncate">Build Session</div>
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
      <div className="flex-1 min-h-0 flex flex-col">
        {/* Model info (collapsible) */}
        <div className="shrink-0 p-3 border-b border-replit-border/40">
          <button
            type="button"
            onClick={() => setModelInfoOpen((v) => !v)}
            className="w-full flex items-center justify-between rounded-xl border border-replit-border bg-replit-bg px-3 py-2 text-sm font-semibold"
            aria-expanded={modelInfoOpen}
          >
            <span>Model info</span>
            {modelInfoOpen ? <ChevronUp className="w-4 h-4 text-replit-textMuted" /> : <ChevronDown className="w-4 h-4 text-replit-textMuted" />}
          </button>

          {modelInfoOpen && (
            <div className="mt-3 rounded-xl border border-replit-border bg-replit-bg p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs text-replit-textMuted">Model</div>
                  <div className="text-sm font-semibold truncate">{session.modelName}</div>
                </div>
                <button
                  onClick={onEditSession}
                  className="text-xs text-replit-accent hover:text-replit-accentHover inline-flex items-center gap-1"
                >
                  Edit <ExternalLink className="w-3 h-3" />
                </button>
              </div>

              <div className="mt-3">
                <div className="text-xs text-replit-textMuted">Goal</div>
                <div className="mt-1 text-sm text-replit-text leading-relaxed">{goalSummary}</div>
              </div>

              <div className="mt-3">
                <div className="text-xs text-replit-textMuted">Datasets</div>
                {datasetLinks.length > 0 ? (
                  <div className="mt-2 space-y-2">
                    {datasetLinks.map((link, idx) => (
                      <div
                        key={`${link}-${idx}`}
                        className="flex items-start gap-2 rounded-lg border border-replit-border bg-replit-surface px-3 py-2"
                      >
                        <div className="min-w-0 flex-1 text-xs text-replit-textMuted break-words">{link}</div>
                        <button
                          type="button"
                          onClick={() => removeDataset(idx)}
                          className="shrink-0 p-1 rounded hover:bg-replit-surfaceHover/40 text-replit-textMuted"
                          aria-label="Remove dataset"
                          title="Remove"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="mt-1 text-sm text-replit-textMuted">Auto dataset discovery</div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Chat history */}
        <div ref={chatScrollRef} className="flex-1 min-h-0 overflow-auto p-4">
          <div className="space-y-3">
            {chatHistory.length === 0 ? (
              <div className="text-sm text-replit-textMuted">No messages yet.</div>
            ) : (
              chatHistory.map((m) => (
                <div
                  key={m.id}
                  className={
                    'max-w-[95%] rounded-2xl border px-4 py-3 text-sm leading-relaxed ' +
                    (m.role === 'user'
                      ? 'ml-auto border-replit-accent/30 bg-replit-accent/10 text-replit-text'
                      : 'mr-auto border-replit-border bg-replit-bg text-replit-text')
                  }
                >
                  <div className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]">{m.text}</div>
                </div>
              ))
            )}

            {aiThinking ? (
              <div className="mr-auto max-w-[95%] rounded-2xl border border-replit-border bg-replit-bg px-4 py-3">
                <MatrixThinkingLoader active />
              </div>
            ) : null}
          </div>
        </div>

        {/* Bottom (pinned) chat input */}
        <div className="shrink-0 border-t border-replit-border/60 bg-replit-surface p-3">
          <div ref={optionsRef} className="relative">
            {isOptionsOpen ? (
              <div className="mb-3 rounded-xl border border-replit-border bg-replit-bg p-3">
                <div className="flex items-center gap-2 flex-wrap">
                  <button
                    type="button"
                    onClick={() => setActiveOption('datasets')}
                    className={
                      'px-3 py-2 rounded-lg text-xs font-semibold border transition-colors ' +
                      (activeOption === 'datasets'
                        ? 'border-replit-accent/70 bg-replit-accent/15 text-replit-text'
                        : 'border-replit-border/60 bg-replit-surface/20 text-replit-textMuted hover:bg-replit-surfaceHover/30')
                    }
                  >
                    Add dataset
                  </button>

                  <button
                    type="button"
                    onClick={() => setActiveOption('constraints')}
                    className={
                      'px-3 py-2 rounded-lg text-xs font-semibold border transition-colors inline-flex items-center gap-2 ' +
                      (activeOption === 'constraints'
                        ? 'border-replit-accent/70 bg-replit-accent/15 text-replit-text'
                        : 'border-replit-border/60 bg-replit-surface/20 text-replit-textMuted hover:bg-replit-surfaceHover/30')
                    }
                  >
                    <SlidersHorizontal className="w-4 h-4" /> Constraints
                  </button>
                </div>

                {activeOption === 'datasets' ? (
                  <div className="mt-3">
                    <div className="relative">
                      <input
                        value={datasetDraft}
                        onChange={(e) => {
                          setDatasetDraft(e.target.value);
                          if (datasetError) setDatasetError(null);
                        }}
                        placeholder="Kaggle dataset link…"
                        className="w-full rounded-xl border border-replit-border bg-replit-bg px-4 py-3 pr-20 text-sm outline-none focus:border-replit-accent/80 focus:ring-2 focus:ring-replit-accent/20 transition"
                      />
                      <button
                        type="button"
                        onClick={addDataset}
                        className={
                          'absolute right-2 top-1/2 -translate-y-1/2 px-3 py-2 rounded-lg text-xs font-semibold transition-colors ' +
                          (datasetDraft.trim()
                            ? 'bg-replit-accent hover:bg-replit-accentHover text-white'
                            : 'bg-replit-surfaceHover text-replit-textMuted border border-replit-border/60 cursor-not-allowed')
                        }
                        disabled={!datasetDraft.trim()}
                      >
                        Add
                      </button>
                    </div>
                    {datasetError ? (
                      <div className="mt-2 text-xs text-replit-warning">{datasetError}</div>
                    ) : (
                      <div className="mt-2 text-xs text-replit-textMuted">Adds to this session’s dataset list.</div>
                    )}
                  </div>
                ) : null}

                {activeOption === 'constraints' ? (
                  <div className="mt-3 space-y-3">
                    <div>
                      <div className="text-xs text-replit-textMuted">Time budget</div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {(['fast', 'balanced', 'thorough'] as const).map((opt) => (
                          <button
                            key={opt}
                            onClick={() => setTimeBudget(opt)}
                            className={
                              'px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors ' +
                              (timeBudget === opt
                                ? 'border-replit-accent/70 bg-replit-accent/15 text-replit-text'
                                : 'border-replit-border/60 bg-replit-surface/20 text-replit-textMuted hover:bg-replit-surfaceHover/30')
                            }
                            type="button"
                          >
                            {opt}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <div className="flex items-center justify-between">
                        <div className="text-xs text-replit-textMuted">Candidate models</div>
                        <div className="text-xs font-mono text-replit-textMuted">{candidates}</div>
                      </div>
                      <input
                        type="range"
                        min={2}
                        max={12}
                        step={1}
                        value={candidates}
                        onChange={(e) => setCandidates(parseInt(e.target.value, 10))}
                        className="w-full mt-2 accent-replit-accent"
                      />
                    </div>
                  </div>
                ) : null}
              </div>
            ) : null}

            <div className="relative">
              <textarea
                ref={messageRef}
                value={messageDraft}
                onChange={(e) => setMessageDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                rows={1}
                placeholder="Message..."
                className="w-full resize-none rounded-xl border border-replit-border bg-replit-bg px-4 pr-16 py-3 text-sm leading-relaxed outline-none focus:border-replit-accent/80 focus:ring-2 focus:ring-replit-accent/20 transition max-h-40 overflow-y-hidden whitespace-pre-wrap break-words [overflow-wrap:anywhere]"
              />

              <div className="absolute right-2 bottom-2 flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => {
                    setIsOptionsOpen((v) => !v);
                    setActiveOption('none');
                  }}
                  className="h-7 w-7 inline-flex items-center justify-center rounded-md text-replit-textMuted hover:text-replit-text hover:bg-replit-surfaceHover/20 transition-colors"
                  aria-label="Options"
                  aria-expanded={isOptionsOpen}
                  title="Options"
                >
                  <Plus className="w-4 h-4" />
                </button>

                <button
                  type="button"
                  onClick={sendMessage}
                  disabled={!messageDraft.trim()}
                  className={
                    'h-7 w-7 inline-flex items-center justify-center rounded-md transition-colors ' +
                    (messageDraft.trim()
                      ? 'text-replit-text hover:bg-replit-surfaceHover/20'
                      : 'text-replit-textMuted cursor-not-allowed')
                  }
                  aria-label="Send"
                  title="Send"
                >
                  <ArrowUp className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
