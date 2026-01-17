import { useState, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, Plus, Moon, Sun } from 'lucide-react';

import {
  createBuildSession,
  isValidKaggleDatasetLink,
  setCurrentSession,
} from '../lib/buildSession';
import { useRouter } from '../router/router';
import { useTheme } from '../lib/theme';
import MatrixScreenLoader from '../components/MatrixScreenLoader';

export default function HomePage() {
  const { navigate } = useRouter();
  const { toggleTheme, theme } = useTheme();

  const [goalPrompt, setGoalPrompt] = useState('');
  const [kaggleLink, setKaggleLink] = useState('');
  const [showDatasetMenu, setShowDatasetMenu] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isBarFocused, setIsBarFocused] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);

  const kaggleOk = useMemo(
    () => !kaggleLink || isValidKaggleDatasetLink(kaggleLink),
    [kaggleLink]
  );

  const canStart = goalPrompt.trim().length > 0;

  const idleGlow =
    '0 0 0 1px rgba(var(--replit-accent-rgb), 0.18), 0 0 22px rgba(var(--replit-accent-rgb), 0.16), 0 0 80px rgba(var(--replit-accent-rgb), 0.06)';
  const idleGlowHover =
    '0 0 0 1px rgba(var(--replit-accent-rgb), 0.32), 0 0 34px rgba(var(--replit-accent-rgb), 0.26), 0 0 110px rgba(var(--replit-accent-rgb), 0.12)';
  const selectedGlow =
    '0 0 0 2px rgba(var(--replit-accent-rgb), 0.62), 0 0 70px rgba(var(--replit-accent-rgb), 0.52), 0 0 160px rgba(var(--replit-accent-rgb), 0.22)';
  const selectedGlowHover =
    '0 0 0 2px rgba(var(--replit-accent-rgb), 0.72), 0 0 90px rgba(var(--replit-accent-rgb), 0.64), 0 0 190px rgba(var(--replit-accent-rgb), 0.28)';

  const onStart = () => {
    if (!canStart) return;

    setIsStarting(true);
    const session = createBuildSession({
      modelName: '',
      goalPrompt,
      kaggleLink,
    });

    setCurrentSession(session);
    navigate('/workspace');
  };

  return (
    <div className="min-h-screen bg-replit-bg text-replit-text relative overflow-hidden">
      {isStarting && <MatrixScreenLoader label="Starting build…" />}

      {/* Logo */}
      <motion.div
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: 'easeOut' }}
        className="absolute top-6 left-6 flex items-center gap-2"
      >
        <div className="text-lg font-semibold tracking-tight">
          <span className="text-replit-text">AI</span>
          <span className="text-replit-accent">AI</span>
        </div>
        <div className="h-2 w-2 rounded-full bg-replit-accent shadow-[0_0_12px_rgba(15,98,254,0.8)]" />
      </motion.div>

      {/* Theme toggle */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="absolute top-6 right-6"
      >
        <button
          onClick={toggleTheme}
          className="rounded-lg border border-replit-border bg-replit-surface p-2 hover:bg-replit-surfaceHover transition"
        >
          {theme === 'midnight' ? <Moon size={16} /> : <Sun size={16} />}
        </button>
      </motion.div>

      {/* Center content */}
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="flex min-h-screen flex-col items-center justify-center px-6"
      >
        <h1 className="text-2xl md:text-3xl font-medium mb-6 text-center">
          Hey, Mohamed. Ready to dive in?
        </h1>

        {/* Prompt container */}
        <motion.div layout className="relative w-full max-w-2xl">
          <motion.div
            style={{
              boxShadow: isBarFocused ? selectedGlow : idleGlow,
            }}
            whileHover={{
              boxShadow:
                isBarFocused
                  ? selectedGlowHover
                  : idleGlowHover,
            }}
            transition={{
              duration: 0.16,
              ease: 'easeOut',
            }}
            onFocusCapture={() => setIsBarFocused(true)}
            onBlurCapture={(e) => {
              const next = e.relatedTarget as Node | null;
              if (!next || !e.currentTarget.contains(next)) {
                setIsBarFocused(false);
              }
            }}
            className="flex items-center rounded-2xl bg-replit-surface border border-replit-border px-3 py-2 shadow-sm"
          >
            {/* + menu */}
            <div className="relative">
              <motion.button
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowDatasetMenu(v => !v)}
                className="p-2 rounded-lg hover:bg-replit-surfaceHover transition"
              >
                <Plus size={18} />
              </motion.button>

              <AnimatePresence>
                {showDatasetMenu && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.96, y: -6 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.96, y: -6 }}
                    transition={{ duration: 0.18, ease: 'easeOut' }}
                    className="absolute left-0 top-11 w-72 rounded-xl border border-replit-border bg-replit-surface p-3 z-20 backdrop-blur"
                  >
                    <label className="text-xs font-medium mb-1 block">
                      Kaggle dataset link
                    </label>
                    <input
                      value={kaggleLink}
                      onChange={(e) => setKaggleLink(e.target.value)}
                      placeholder="https://kaggle.com/datasets/..."
                      className={
                        'w-full rounded-lg bg-replit-bg border px-3 py-2 text-sm outline-none transition ' +
                        (kaggleOk
                          ? 'border-replit-border focus:ring-2 focus:ring-replit-accent/30'
                          : 'border-yellow-400 focus:ring-2 focus:ring-yellow-400/30')
                      }
                    />
                    <p className="mt-1 text-xs text-replit-textMuted">
                      Leave blank to auto-select a dataset
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Prompt input */}
            <input
              ref={inputRef}
              value={goalPrompt}
              onChange={(e) => setGoalPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && onStart()}
              placeholder="Ask anything"
              className="flex-1 bg-transparent px-3 py-2 text-sm outline-none"
            />

            {/* Send button */}
            <motion.button
              whileHover={canStart ? { scale: 1.05 } : undefined}
              whileTap={canStart ? { scale: 0.95 } : undefined}
              onClick={onStart}
              disabled={!canStart}
              className={
                'p-2 rounded-lg transition ' +
                (canStart
                  ? 'bg-replit-accent text-white'
                  : 'text-replit-textMuted')
              }
            >
              <ArrowRight size={18} />
            </motion.button>
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
}
