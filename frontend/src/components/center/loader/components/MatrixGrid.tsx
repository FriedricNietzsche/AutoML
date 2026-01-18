import { useMemo } from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { seeded, clamp01 } from '../types';

interface MatrixGridProps {
  label: string;
  rows: number;
  cols: number;
  timeMs: number;
  reducedMotion: boolean;
}

/**
 * Animated matrix grid visualization
 * Shows a grid of cells with dynamic values and highlighting for operations
 */
export default function MatrixGrid({
  label,
  rows,
  cols,
  timeMs,
  reducedMotion,
}: MatrixGridProps) {
  const cells = useMemo(() => {
    const list: number[] = [];
    const total = rows * cols;
    for (let i = 0; i < total; i += 1) list.push(seeded(i + rows * 13 + cols * 7));
    return list;
  }, [rows, cols]);

  const opIndex = reducedMotion ? 0 : Math.floor(timeMs / 110) % Math.max(1, rows * cols);
  const opRow = Math.floor(opIndex / cols);
  const opCol = opIndex % cols;

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-mono text-replit-textMuted">
          {label}{' '}
          <span className="px-1.5 py-0.5 rounded border border-replit-border/60 bg-replit-surface/40">
            {rows}×{cols}
          </span>
        </div>
        <div className="text-[11px] text-replit-textMuted">Matrix op</div>
      </div>

      <div
        className="grid gap-px rounded-lg border border-replit-border/60 bg-replit-border/60 p-px overflow-hidden"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {cells.map((val, idx) => {
          const r = Math.floor(idx / cols);
          const c = idx % cols;

          const inOpRow = r === opRow;
          const inOpCol = c === opCol;
          const isOpCell = inOpRow && inOpCol;

          const wave = reducedMotion ? 0 : Math.sin(timeMs / 240 + idx * 0.35) * 0.08;
          const opBoost = reducedMotion ? 0 : inOpRow || inOpCol ? 0.12 : 0;
          const hotBoost = reducedMotion ? 0 : isOpCell ? 0.18 : 0;

          const dyn = clamp01(val + wave + opBoost + hotBoost);
          const display = (dyn * 2 - 1) * 1.15;

          return (
            <motion.div
              key={idx}
              initial={false}
              animate={
                reducedMotion
                  ? undefined
                  : {
                      opacity: 1,
                      scale: isOpCell ? 1.02 : 1,
                    }
              }
              transition={reducedMotion ? undefined : { type: 'spring', stiffness: 320, damping: 28 }}
              className={clsx(
                'flex items-center justify-center font-mono select-none',
                'h-10 md:h-12',
                'text-xs md:text-sm',
                'bg-replit-surface/40 text-replit-text',
                (inOpRow || inOpCol) && 'bg-replit-surfaceHover/60 ring-1 ring-replit-accent/20',
                isOpCell && 'ring-2 ring-replit-accent/70'
              )}
              style={{ opacity: 0.5 + dyn * 0.4 }}
            >
              {display.toFixed(2)}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
