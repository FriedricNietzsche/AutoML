import clsx from 'clsx';
import type { VisualProps } from '../types';
import { clamp01 } from '../types';

function buildFallbackConfusion(size: number) {
  return Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) => (r === c ? 45 + r * 6 : 8 + c * 3))
  );
}

export default function ConfusionMatrixViz({ confusion, phaseProgress, reducedMotion }: VisualProps) {
  const confusionFinal = confusion?.length ? confusion : buildFallbackConfusion(3);
  const classes = confusionFinal.length;
  const totalCells = classes * classes;
  const p = reducedMotion ? 1 : clamp01(phaseProgress);

  const flat = confusionFinal.flat();
  const maxValue = flat.length ? Math.max(...flat) : 1;

  const cellTarget = clamp01(p);
  const confusionVisible = Array.from({ length: classes }, (_, r) =>
    Array.from({ length: classes }, (_, c) => {
      const idx = r * classes + c;
      const start = idx / totalCells;
      const end = (idx + 1) / totalCells;
      const local = clamp01((cellTarget - start) / Math.max(1e-9, end - start));
      const final = confusionFinal[r][c];
      return reducedMotion ? final : Math.min(final, Math.floor(final * local));
    })
  );

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-replit-border/60 bg-replit-surface/35 backdrop-blur p-6">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold text-replit-text">Confusion Matrix</div>
            <div className="text-sm text-replit-textMuted">Large matrix view</div>
          </div>
          <div className="text-xs font-mono text-replit-textMuted">{classes}×{classes}</div>
        </div>

        <div className="rounded-xl border border-replit-border/60 bg-replit-bg p-6 overflow-hidden">
          <div className="flex items-center justify-center overflow-hidden">
            <div className="inline-block rounded-lg p-4 border border-replit-border/60 bg-replit-surface/30 max-h-[560px] overflow-auto">
              <div className="mb-3 text-center text-replit-textMuted text-sm">Predicted Class</div>
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center" style={{ width: '90px' }}>
                  <span className="text-replit-textMuted font-semibold transform -rotate-90 whitespace-nowrap">Actual Class</span>
                </div>
                <table className="border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2"></th>
                      {confusionVisible.map((_, idx) => (
                        <th key={idx} className="p-2 text-replit-text font-semibold">
                          {idx}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionVisible.map((row, i) => (
                      <tr key={i}>
                        <td className="p-2 text-replit-text font-semibold">{i}</td>
                        {row.map((cell, j) => {
                          const isDiag = i === j;
                          return (
                            <td
                              key={j}
                              className={clsx(
                                'text-center font-mono font-bold border border-replit-border/60',
                                'text-replit-text',
                                isDiag && 'ring-1 ring-replit-success/40'
                              )}
                              style={{
                                backgroundColor: `rgba(99,102,241,${Math.min(0.9, 0.2 + (cell / Math.max(1, maxValue)) * 0.7)})`,
                                width: '64px',
                                height: '64px',
                                padding: '0',
                              }}
                            >
                              {cell}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
