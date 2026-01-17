import { Brain, Sparkles } from 'lucide-react';
import { motion, useReducedMotion } from 'framer-motion';

export default function EmptyPreview() {
  const reduceMotion = useReducedMotion();

  return (
    <div className="h-full flex flex-col items-center justify-center p-8 text-center bg-replit-bg">
      <motion.div
        initial={reduceMotion ? false : { scale: 0.98, opacity: 0, y: 10 }}
        animate={reduceMotion ? undefined : { scale: 1, opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
        className="max-w-md w-full"
      >
        <div className="rounded-2xl border border-replit-border bg-replit-surface shadow-sm p-8 relative overflow-hidden">
          {/* Decorative neural network lines */}
          <div className="absolute inset-0 opacity-25 pointer-events-none">
            <svg viewBox="0 0 400 260" className="w-full h-full">
              <defs>
                <linearGradient id="nn" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0" stopColor="rgb(var(--replit-accent))" stopOpacity="0.35" />
                  <stop offset="1" stopColor="rgb(var(--replit-warning))" stopOpacity="0.25" />
                </linearGradient>
              </defs>
              {[
                [60, 60], [60, 130], [60, 200],
                [200, 40], [200, 100], [200, 160], [200, 220],
                [340, 80], [340, 150], [340, 210],
              ].map(([x, y], i) => (
                <circle key={i} cx={x} cy={y} r="6" fill="url(#nn)" />
              ))}
              {[
                [60, 60, 200, 40], [60, 60, 200, 100], [60, 60, 200, 160], [60, 60, 200, 220],
                [60, 130, 200, 40], [60, 130, 200, 100], [60, 130, 200, 160], [60, 130, 200, 220],
                [60, 200, 200, 40], [60, 200, 200, 100], [60, 200, 160, 220], [60, 200, 200, 220],
                [200, 40, 340, 80], [200, 40, 340, 150], [200, 40, 340, 210],
                [200, 100, 340, 80], [200, 100, 340, 150], [200, 100, 340, 210],
                [200, 160, 340, 80], [200, 160, 340, 150], [200, 160, 340, 210],
                [200, 220, 340, 80], [200, 220, 340, 150], [200, 220, 340, 210],
              ].map(([x1, y1, x2, y2], i) => (
                <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="url(#nn)" strokeWidth="1" />
              ))}
            </svg>
          </div>

          <div className="relative z-10 flex flex-col items-center">
            <div className="w-14 h-14 rounded-2xl bg-replit-bg border border-replit-border flex items-center justify-center mb-5">
              <Brain className="w-7 h-7 text-replit-accent" />
            </div>

            <h2 className="text-2xl font-bold">Build something</h2>
            <p className="mt-2 text-sm text-replit-textMuted leading-relaxed">
              Start a build session on the Home page, then come back here to watch the sequential training simulator.
            </p>

            <div className="mt-6 w-full rounded-xl border border-replit-border bg-replit-bg p-4 text-left">
              <div className="flex items-start gap-3">
                <div className="p-2 rounded-lg bg-replit-surface border border-replit-border">
                  <Sparkles className="w-4 h-4 text-replit-textMuted" />
                </div>
                <div>
                  <div className="text-xs font-bold uppercase tracking-wider text-replit-textMuted">Fun AI fact</div>
                  <div className="mt-1 text-xs text-replit-textMuted leading-relaxed">
                    Transformers (used by GPT-style models) were introduced in 2017 and replaced many RNN-based approaches for text.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
