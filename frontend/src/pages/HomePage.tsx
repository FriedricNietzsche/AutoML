import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Sparkles, ArrowRight, Moon, Sun } from 'lucide-react';
import { createBuildSession, isValidKaggleDatasetLink, setCurrentSession } from '../lib/buildSession';
import { useRouter } from '../router/router';
import { useTheme } from '../lib/theme';
import MatrixScreenLoader from '../components/MatrixScreenLoader';

function MatrixMiniLoader({ active }: { active: boolean }) {
  const reduceMotion = useReducedMotion();
  if (!active) return null;

  return (
    <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2">
      <div
        className={
          reduceMotion
            ? 'grid grid-cols-3 gap-0.5 opacity-50'
            : 'grid grid-cols-3 gap-0.5 opacity-70'
        }
        aria-hidden
      >
        {Array.from({ length: 9 }).map((_, i) => (
          <div
            key={i}
            className={
              reduceMotion
                ? 'h-1.5 w-1.5 rounded-[2px] bg-replit-textMuted/40'
                : 'h-1.5 w-1.5 rounded-[2px] bg-replit-textMuted/40 animate-[matrixPulse_900ms_ease-in-out_infinite]'
            }
            style={!reduceMotion ? { animationDelay: `${i * 80}ms` } : undefined}
          />
        ))}
      </div>
    </div>
  );
}

function useTypingValidation(delayMs: number) {
  const [isValidating, setIsValidating] = useState(false);
  const timerRef = useRef<number | null>(null);

  const trigger = () => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
    setIsValidating(true);
    timerRef.current = window.setTimeout(() => setIsValidating(false), delayMs);
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, []);

  return { isValidating, trigger };
}

export default function HomePage() {
  const { navigate } = useRouter();
  const { toggleTheme, theme } = useTheme();
  const reduceMotion = useReducedMotion();

  const [modelName, setModelName] = useState('');
  const [goalPrompt, setGoalPrompt] = useState('');
  const [kaggleLink, setKaggleLink] = useState('');
  const [isStarting, setIsStarting] = useState(false);

  const nameValidation = useTypingValidation(420);
  const promptValidation = useTypingValidation(420);
  const kaggleValidation = useTypingValidation(420);

  const kaggleOk = useMemo(() => {
    if (!kaggleLink.trim()) return false;
    return isValidKaggleDatasetLink(kaggleLink);
  }, [kaggleLink]);

  const kaggleHint = useMemo(() => {
    if (!kaggleLink.trim()) return 'Auto dataset discovery will be used.';
    if (!kaggleOk) return 'Auto dataset discovery will be used.';
    return 'Dataset link looks valid.';
  }, [kaggleLink, kaggleOk]);

  const onStartBuild = () => {
    setIsStarting(true);
    const session = createBuildSession({ modelName, goalPrompt, kaggleLink });
    setCurrentSession(session);
    navigate('/workspace');
  };

  const canStart = goalPrompt.trim().length > 0;

  return (
    <div className="min-h-screen bg-replit-bg text-replit-text relative overflow-hidden">
      {isStarting ? <MatrixScreenLoader label="Starting build…" /> : null}
      {/* Background */}
      <div className="absolute inset-0 -z-10">
        <div className="home-bg-static" />
      </div>

      <div className="max-w-5xl mx-auto px-6 py-14">
        {/* Header */}
        <div className="flex items-start justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-replit-border bg-replit-surface px-3 py-1 text-xs text-replit-textMuted">
              <Sparkles className="w-3.5 h-3.5" />
              AutoAI Builder
            </div>
            <h1 className="mt-4 text-4xl md:text-5xl font-semibold tracking-tight">
              Build an AI model UI-first.
            </h1>
            <p className="mt-3 text-replit-textMuted max-w-xl">
              No backend required. Start a build session and watch the training simulator run.
            </p>
          </div>

          <button
            onClick={toggleTheme}
            className="rounded-lg border border-replit-border bg-replit-surface px-3 py-2 text-xs text-replit-text hover:bg-replit-surfaceHover transition-colors"
            aria-label="Toggle theme"
            title={theme === 'midnight' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'midnight' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
          </button>
        </div>

        {/* Form Card */}
        <motion.div
          initial={reduceMotion ? false : { opacity: 0, y: 10 }}
          animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
          style={{ willChange: 'transform, opacity' }}
          className="mt-10 mx-auto max-w-2xl"
        >
          <div className="rounded-2xl border border-replit-border bg-replit-surface shadow-sm">
            <div className="p-6 md:p-7 border-b border-replit-border/40">
              <h2 className="text-lg font-semibold">Start a Build Session</h2>
              <p className="mt-1 text-sm text-replit-textMuted">
                Tell AutoAI what to build. We’ll simulate the pipeline and generate artifacts into the VFS.
              </p>
            </div>

            <div className="p-6 md:p-7 space-y-5">
              {/* Model Name */}
              <div>
                <label className="text-sm font-medium">Model Name</label>
                <div className="mt-2 relative">
                  <input
                    value={modelName}
                    onChange={(e) => {
                      setModelName(e.target.value);
                      nameValidation.trigger();
                    }}
                    placeholder="e.g. Review Sentiment Classifier"
                    className="w-full rounded-xl border border-replit-border bg-replit-bg px-4 py-3 pr-10 text-sm outline-none focus:border-replit-accent/80 focus:ring-2 focus:ring-replit-accent/20 transition"
                  />
                  <MatrixMiniLoader active={nameValidation.isValidating} />
                </div>
              </div>

              {/* Goal Prompt */}
              <div>
                <label className="text-sm font-medium">What should the model do?</label>
                <div className="mt-2 relative">
                  <textarea
                    value={goalPrompt}
                    onChange={(e) => {
                      setGoalPrompt(e.target.value);
                      promptValidation.trigger();
                    }}
                    rows={4}
                    placeholder="Describe the task and expected inputs/outputs…"
                    className="w-full resize-none rounded-xl border border-replit-border bg-replit-bg px-4 py-3 pr-10 text-sm outline-none focus:border-replit-accent/80 focus:ring-2 focus:ring-replit-accent/20 transition"
                  />
                  <MatrixMiniLoader active={promptValidation.isValidating} />
                </div>
                {!goalPrompt.trim() && (
                  <div className="mt-2 text-xs text-replit-textMuted">
                    Required. This drives the simulated API schema and demo inference.
                  </div>
                )}
              </div>

              {/* Kaggle */}
              <div>
                <label className="text-sm font-medium">Kaggle dataset link (optional)</label>
                <div className="mt-2 relative">
                  <input
                    value={kaggleLink}
                    onChange={(e) => {
                      setKaggleLink(e.target.value);
                      kaggleValidation.trigger();
                    }}
                    placeholder="https://kaggle.com/datasets/..."
                    className={
                      'w-full rounded-xl border bg-replit-bg px-4 py-3 pr-10 text-sm outline-none transition ' +
                      (kaggleLink.trim() && !kaggleOk
                        ? 'border-yellow-400/60 focus:border-yellow-400 focus:ring-2 focus:ring-yellow-400/20'
                        : 'border-replit-border focus:border-replit-accent/80 focus:ring-2 focus:ring-replit-accent/20')
                    }
                  />
                  <MatrixMiniLoader active={kaggleValidation.isValidating} />
                </div>
                <div className="mt-2 text-xs text-replit-textMuted">
                  Paste a Kaggle dataset link. Leave blank and AutoAI will choose one.
                </div>
                <div className={
                  'mt-1 text-xs ' +
                  (kaggleLink.trim() && !kaggleOk ? 'text-yellow-300' : kaggleOk ? 'text-green-300' : 'text-replit-textMuted')
                }>
                  {kaggleHint}
                </div>
              </div>

              {/* CTA */}
              <div className="pt-2 flex items-center justify-between gap-4">
                <div className="text-xs text-replit-textMuted">
                  Build starts automatically in the workspace.
                </div>

                <button
                  onClick={onStartBuild}
                  disabled={!canStart}
                  className={
                    'inline-flex items-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition ' +
                    (canStart
                      ? 'bg-replit-accent hover:bg-replit-accentHover text-white shadow-[0_10px_30px_rgba(15,98,254,0.35)]'
                      : 'bg-replit-surfaceHover text-replit-textMuted border border-replit-border/60 cursor-not-allowed')
                  }
                >
                  Start Build
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
