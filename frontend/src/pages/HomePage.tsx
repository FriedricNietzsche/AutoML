import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { ArrowRight, Link as LinkIcon, Moon, Plus, Sparkles, Sun, Upload, X } from 'lucide-react';
import { createBuildSession, isValidKaggleDatasetLink, setCurrentSession } from '../lib/buildSession';
import { useRouter } from '../router/router';
import { useTheme } from '../lib/theme';
import MatrixScreenLoader from '../components/MatrixScreenLoader';

const DATASET_OPTIONS = [
  { value: 'auto', label: 'Auto dataset' },
  { value: 'kaggle', label: 'Kaggle link' },
  { value: 'csv', label: 'Upload file' },
];

type DatasetChoice = (typeof DATASET_OPTIONS)[number]['value'];

export default function HomePage() {
  const { navigate } = useRouter();
  const { toggleTheme, theme } = useTheme();
  const reduceMotion = useReducedMotion();

  // Avoid a "two-step" enter animation on hard refresh in React StrictMode (dev).
  // StrictMode intentionally mounts/unmounts/remounts components, which can replay initial animations.
  const [enableEnterAnim] = useState(() => {
    if (typeof window === 'undefined') return false;
    try {
      return window.sessionStorage.getItem('home.entered') !== '1';
    } catch {
      return true;
    }
  });

  useEffect(() => {
    try {
      window.sessionStorage.setItem('home.entered', '1');
    } catch {
      // ignore
    }
  }, []);

  const [goalPrompt, setGoalPrompt] = useState('');
  const [datasetChoice, setDatasetChoice] = useState<DatasetChoice>(DATASET_OPTIONS[0].value);
  const [datasetMenuOpen, setDatasetMenuOpen] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [kaggleLink, setKaggleLink] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const datasetMenuRef = useRef<HTMLDivElement | null>(null);

  const datasetLabel = useMemo(() => {
    if (datasetChoice === 'csv' && uploadFile) return `Upload: ${uploadFile.name}`;
    return DATASET_OPTIONS.find((opt) => opt.value === datasetChoice)?.label ?? 'Auto dataset';
  }, [datasetChoice, uploadFile]);

  const kaggleOk = useMemo(() => {
    if (datasetChoice !== 'kaggle') return true;
    return isValidKaggleDatasetLink(kaggleLink);
  }, [datasetChoice, kaggleLink]);

  const datasetOk = useMemo(() => {
    if (datasetChoice === 'auto') return true;
    if (datasetChoice === 'kaggle') return kaggleOk;
    if (datasetChoice === 'csv') return !!uploadFile;
    return true;
  }, [datasetChoice, kaggleOk, uploadFile]);

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      if (!datasetMenuOpen) return;
      const target = event.target as Node | null;
      if (!target) return;
      if (datasetMenuRef.current && datasetMenuRef.current.contains(target)) return;
      setDatasetMenuOpen(false);
    };
    window.addEventListener('pointerdown', onPointerDown);
    return () => window.removeEventListener('pointerdown', onPointerDown);
  }, [datasetMenuOpen]);

  const onStartBuild = () => {
    if (!goalPrompt.trim() || isStarting) return;
    if (!datasetOk) return;

    setIsStarting(true);

    requestAnimationFrame(() => {
      const session = createBuildSession({
        modelName: `AutoML (${datasetLabel})`,
        goalPrompt,
        kaggleLink: datasetChoice === 'kaggle' ? kaggleLink.trim() : undefined,
        datasetLinks:
          datasetChoice === 'csv' && uploadFile
            ? [`upload:${uploadFile.name}`]
            : undefined,
      });

      setCurrentSession(session);

      setTimeout(() => navigate('/workspace'), 120);
    });
  };

  const canStart = goalPrompt.trim().length > 0 && datasetOk;

  return (
    <div className="min-h-screen bg-replit-bg text-replit-text relative overflow-hidden">
      {isStarting && <MatrixScreenLoader label="Starting build…" />}

      {/* Static background */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="home-bg-static" />
        <div className="absolute -left-24 -top-24 h-72 w-72 rounded-full bg-replit-accent/15 blur-[80px]" />
        <div className="absolute -right-32 top-24 h-96 w-96 rounded-full bg-purple-500/10 blur-[90px]" />
      </div>

      <div className="mx-auto flex min-h-screen w-full flex-col px-6 py-10">
        {/* Top bar */}
        <div className="flex items-center justify-between">
          <motion.div
            initial={!reduceMotion && enableEnterAnim ? { opacity: 0, y: -4 } : false}
            animate={{ opacity: 1, y: 0 }}
            transition={!reduceMotion && enableEnterAnim ? { duration: 0.25, ease: 'easeOut' } : undefined}
            style={{ willChange: 'transform, opacity' }}
            className="inline-flex items-center gap-2 rounded-full border border-replit-border bg-replit-surface px-3 py-1 text-xs text-replit-textMuted"
          >
            <Sparkles className="h-3.5 w-3.5" />
            AutoML
          </motion.div>

          <motion.button
            onClick={toggleTheme}
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            transition={{ duration: 0.12, ease: 'easeOut' }}
            className="rounded-lg border border-replit-border bg-replit-surface/60 backdrop-blur px-3 py-2 text-xs"
          >
            {theme === 'midnight' ? (
              <Moon className="w-4 h-4" />
            ) : (
              <Sun className="w-4 h-4" />
            )}
          </motion.button>
        </div>

        {/* Main content */}
        <motion.div
          initial={!reduceMotion && enableEnterAnim ? { opacity: 0, y: 10 } : false}
          animate={{ opacity: 1, y: 0 }}
          transition={!reduceMotion && enableEnterAnim ? { duration: 0.3, ease: 'easeOut' } : undefined}
          style={{ willChange: 'transform, opacity' }}
          className="flex flex-1 flex-col items-center justify-center"
        >
          <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-center bg-gradient-to-r from-replit-text via-replit-text to-replit-textMuted bg-clip-text text-transparent">
            What’s on the agenda today?
          </h1>

          <p className="mt-2 text-sm text-replit-textMuted text-center max-w-lg">
            Describe the model you want, then add a dataset source.
          </p>

          <div className="mt-8 w-full max-w-4xl">
            {/* Card — NO hover scaling */}
            <div className="rounded-[28px] border border-replit-border bg-replit-surface/80 shadow-[0_12px_40px_rgba(0,0,0,0.18)] ring-1 ring-white/5">
              <div className="px-5 py-4">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                  {/* Dataset selector */}
                  <div ref={datasetMenuRef} className="relative flex items-center gap-2">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    transition={{ duration: 0.12, ease: 'easeOut' }}
                    onClick={() => setDatasetMenuOpen((v) => !v)}
                    className="flex h-9 w-9 items-center justify-center rounded-full border border-replit-border bg-replit-surface text-replit-textMuted"
                    aria-label="Add dataset"
                    title="Add dataset"
                  >
                    <Plus className="h-4 w-4" />
                  </motion.button>

                  <span className="text-xs font-medium text-replit-textMuted">
                    {datasetLabel}
                  </span>

                  <AnimatePresence>
                    {datasetMenuOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -2 }}
                        transition={{ duration: 0.12, ease: 'easeOut' }}
                        style={{ willChange: 'transform, opacity' }}
                        className="absolute left-0 top-11 z-20 w-44 overflow-hidden rounded-xl border border-replit-border bg-replit-surface shadow-lg"
                      >
                        <button
                          type="button"
                          onClick={() => {
                            setDatasetChoice('auto');
                            setDatasetMenuOpen(false);
                            setKaggleLink('');
                            setUploadFile(null);
                          }}
                          className={
                            'flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-replit-surfaceHover ' +
                            (datasetChoice === 'auto' ? 'text-replit-text' : 'text-replit-textMuted')
                          }
                        >
                          <Sparkles className="h-4 w-4" />
                          Auto dataset
                        </button>

                        <button
                          type="button"
                          onClick={() => {
                            setDatasetChoice('kaggle');
                            setDatasetMenuOpen(false);
                            setUploadFile(null);
                          }}
                          className={
                            'flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-replit-surfaceHover ' +
                            (datasetChoice === 'kaggle' ? 'text-replit-text' : 'text-replit-textMuted')
                          }
                        >
                          <LinkIcon className="h-4 w-4" />
                          Kaggle link
                        </button>

                        <button
                          type="button"
                          onClick={() => {
                            setDatasetChoice('csv');
                            setDatasetMenuOpen(false);
                            setKaggleLink('');
                            requestAnimationFrame(() => fileInputRef.current?.click());
                          }}
                          className={
                            'flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-replit-surfaceHover ' +
                            (datasetChoice === 'csv' ? 'text-replit-text' : 'text-replit-textMuted')
                          }
                        >
                          <Upload className="h-4 w-4" />
                          Upload file
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                  {/* Prompt */}
                  <div className="flex-1">
                    <textarea
                      value={goalPrompt}
                      onChange={(e) => setGoalPrompt(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          onStartBuild();
                        }
                      }}
                      rows={1}
                      placeholder="Ask anything"
                      className="w-full resize-none bg-transparent text-sm outline-none leading-[44px] h-11"
                    />
                  </div>

                  {/* Start button */}
                  <motion.button
                    whileHover={canStart ? { scale: 1.04 } : undefined}
                    whileTap={canStart ? { scale: 0.96 } : undefined}
                    transition={{ duration: 0.12, ease: 'easeOut' }}
                    onClick={onStartBuild}
                    disabled={!canStart || isStarting}
                    className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold ${
                      canStart
                        ? 'bg-replit-accent text-white shadow-lg'
                        : 'bg-replit-surfaceHover text-replit-textMuted border border-replit-border cursor-not-allowed'
                    }`}
                    aria-label="Start build"
                  >
                    <ArrowRight className="h-4 w-4" />
                  </motion.button>
                </div>

                {/* Dataset details */}
                {datasetChoice === 'kaggle' ? (
                  <div className="mt-3 border-t border-replit-border/60 pt-3">
                    <div className="relative">
                      <input
                        value={kaggleLink}
                        onChange={(e) => setKaggleLink(e.target.value)}
                        placeholder="https://kaggle.com/datasets/..."
                        className={
                          'w-full rounded-xl border bg-replit-bg px-4 py-3 text-sm outline-none transition ' +
                          (kaggleLink.trim() && !kaggleOk
                            ? 'border-yellow-400/60 focus:border-yellow-400 focus:ring-2 focus:ring-yellow-400/20'
                            : 'border-replit-border focus:border-replit-accent/80 focus:ring-2 focus:ring-replit-accent/20')
                        }
                      />
                    </div>
                    <div className={
                      'mt-2 text-xs ' +
                      (kaggleLink.trim() && !kaggleOk ? 'text-yellow-300' : 'text-replit-textMuted')
                    }>
                      {kaggleLink.trim() && !kaggleOk ? 'Paste a valid Kaggle dataset link to continue.' : 'Paste a Kaggle dataset link.'}
                    </div>
                  </div>
                ) : null}

                {datasetChoice === 'csv' ? (
                  <div className="mt-3 border-t border-replit-border/60 pt-3 flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-xs text-replit-textMuted">Selected file</div>
                      <div className="text-sm text-replit-text truncate">
                        {uploadFile ? uploadFile.name : 'No file selected'}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="rounded-xl border border-replit-border bg-replit-bg px-3 py-2 text-xs font-semibold text-replit-textMuted hover:bg-replit-surfaceHover transition-colors"
                      >
                        Choose file
                      </button>
                      {uploadFile ? (
                        <button
                          type="button"
                          onClick={() => setUploadFile(null)}
                          className="h-9 w-9 inline-flex items-center justify-center rounded-xl border border-replit-border bg-replit-bg text-replit-textMuted hover:bg-replit-surfaceHover transition-colors"
                          aria-label="Clear file"
                          title="Clear file"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      ) : null}
                    </div>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="mt-3 text-center text-xs text-replit-textMuted">
              Tip: Press Enter to start. Shift + Enter adds a new line.
            </div>
          </div>
        </motion.div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.tsv,.json,.txt"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0] ?? null;
          setUploadFile(file);
        }}
      />
    </div>
  );
}
