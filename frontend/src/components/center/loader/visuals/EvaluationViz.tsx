import { useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import {
  Line,
  LineChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { VisualProps } from '../types';
import { clamp01, lerp, seeded } from '../types';

type Metric = {
  key: string;
  label: string;
  value: number; // 0..1
};

type CurvePt = { x: number; y: number };

function buildConfusion(classCount: number, seed: number) {
  const s = seed * 113 + 404;
  const total = 220 + Math.floor(seeded(s + 1) * 220);

  // Start with diagonal-heavy matrix.
  const mat: number[][] = Array.from({ length: classCount }, () => Array.from({ length: classCount }, () => 0));

  let remaining = total;
  for (let i = 0; i < classCount; i += 1) {
    const diag = Math.floor((0.55 + seeded(s + 20 + i) * 0.25) * (total / classCount));
    mat[i][i] = diag;
    remaining -= diag;
  }

  // Distribute remaining across off-diagonals.
  const slots: Array<[number, number]> = [];
  for (let r = 0; r < classCount; r += 1) {
    for (let c = 0; c < classCount; c += 1) {
      if (r !== c) slots.push([r, c]);
    }
  }

  for (let k = 0; k < slots.length; k += 1) {
    if (remaining <= 0) break;
    const [r, c] = slots[k];
    const portion = Math.min(remaining, Math.floor(seeded(s + 300 + k * 7) * (total * 0.08)));
    mat[r][c] += portion;
    remaining -= portion;
  }

  // Put any leftovers anywhere.
  if (remaining > 0) {
    mat[0][classCount - 1] += remaining;
  }

  return mat;
}

function computeMetricsFromConfusion(mat: number[][]) {
  const k = mat.length;
  const total = mat.flat().reduce((a, b) => a + b, 0);
  const tp = Array.from({ length: k }, (_, i) => mat[i][i]);
  const rowSum = Array.from({ length: k }, (_, r) => mat[r].reduce((a, b) => a + b, 0));
  const colSum = Array.from({ length: k }, (_, c) => mat.reduce((a, row) => a + row[c], 0));

  const precisionPer = tp.map((t, i) => t / Math.max(1, colSum[i]));
  const recallPer = tp.map((t, i) => t / Math.max(1, rowSum[i]));
  const f1Per = tp.map((_, i) => {
    const p = precisionPer[i];
    const r = recallPer[i];
    return (2 * p * r) / Math.max(1e-9, p + r);
  });

  const macroPrecision = precisionPer.reduce((a, b) => a + b, 0) / k;
  const macroRecall = recallPer.reduce((a, b) => a + b, 0) / k;
  const macroF1 = f1Per.reduce((a, b) => a + b, 0) / k;
  const accuracy = tp.reduce((a, b) => a + b, 0) / Math.max(1, total);

  return {
    accuracy,
    precision: macroPrecision,
    recall: macroRecall,
    f1: macroF1,
  };
}

function buildCurve(kind: 'pr' | 'roc'): CurvePt[] {
  const pts: CurvePt[] = [];
  const n = 60;
  for (let i = 0; i < n; i += 1) {
    const x = i / (n - 1);
    let y: number;
    if (kind === 'pr') {
      // Match example: precision decays as recall increases.
      y = clamp01(0.95 - x * 0.3);
    } else {
      // Match example: TPR rises above diagonal.
      y = clamp01(x + 0.3 + (1 - x) * 0.5);
    }
    pts.push({ x, y });
  }
  return pts;
}

function computeAuc(points: CurvePt[]) {
  if (points.length < 2) return 0;
  const sorted = [...points].sort((a, b) => a.x - b.x);
  let area = 0;
  for (let i = 1; i < sorted.length; i += 1) {
    const prev = sorted[i - 1];
    const next = sorted[i];
    area += (next.x - prev.x) * (prev.y + next.y) * 0.5;
  }
  return clamp01(area);
}

export default function EvaluationViz({ timeMs, phaseProgress, seed, reducedMotion, writeArtifact, confusion, metrics, curve, showConfusion }: VisualProps) {
  const [classCount, setClassCount] = useState<2 | 3>(3);
  const showMatrix = showConfusion ?? true;

  useEffect(() => {
    if (confusion && (confusion.length === 2 || confusion.length === 3)) {
      setClassCount(confusion.length as 2 | 3);
    }
  }, [confusion]);

  const { confusionFinal, metricsFinal, prCurveFull, rocCurveFull } = useMemo(() => {
    const confusionFinal = confusion?.length
      ? confusion
      : showMatrix
        ? buildConfusion(classCount, seed)
        : [[0]];
    const derived = computeMetricsFromConfusion(confusionFinal);

    const prCurveFull =
      curve?.kind === 'pr'
        ? curve.points
        : buildCurve('pr');
    const rocCurveFull =
      curve?.kind === 'roc'
        ? curve.points
        : buildCurve('roc');

    const auprc = computeAuc(prCurveFull);
    const auroc = computeAuc(rocCurveFull);

    const metricsFinal: Metric[] = metrics
      ? [
          { key: 'accuracy', label: 'Accuracy', value: metrics.accuracy },
          { key: 'precision', label: 'Precision', value: metrics.precision },
          { key: 'recall', label: 'Recall', value: metrics.recall },
          { key: 'f1', label: 'F1 Score', value: metrics.f1 },
          { key: 'auroc', label: 'AUROC', value: auroc },
          { key: 'auprc', label: 'AUPRC', value: auprc },
        ]
      : [
          { key: 'accuracy', label: 'Accuracy', value: derived.accuracy },
          { key: 'precision', label: 'Precision', value: derived.precision },
          { key: 'recall', label: 'Recall', value: derived.recall },
          { key: 'f1', label: 'F1 Score', value: derived.f1 },
          { key: 'auroc', label: 'AUROC', value: auroc },
          { key: 'auprc', label: 'AUPRC', value: auprc },
        ];

    return { confusionFinal, metricsFinal, prCurveFull, rocCurveFull };
  }, [classCount, confusion, curve, metrics, seed, showMatrix]);

  const p = reducedMotion ? 1 : clamp01(phaseProgress);
  const t = reducedMotion ? 1 : timeMs / 1000;

  // Confusion matrix fills cell-by-cell.
  const effectiveClasses = confusionFinal.length;
  const totalCells = effectiveClasses * effectiveClasses;
  const cellTarget = clamp01((p - 0.05) / 0.55);

  const confusionVisible = useMemo(() => {
    const out: number[][] = Array.from({ length: effectiveClasses }, () => Array.from({ length: effectiveClasses }, () => 0));

    for (let idx = 0; idx < totalCells; idx += 1) {
      const r = Math.floor(idx / effectiveClasses);
      const c = idx % effectiveClasses;
      const final = confusionFinal[r][c];

      if (reducedMotion) {
        out[r][c] = final;
        continue;
      }

      const start = idx / totalCells;
      const end = (idx + 1) / totalCells;
      const local = clamp01((cellTarget - start) / Math.max(1e-9, end - start));

      // Count-up with a slight wobble.
      const wobble = 0.02 * Math.sin(t * 4.2 + idx);
      const v = Math.floor(final * clamp01(local + wobble));
      out[r][c] = clamp01(local) <= 0 ? 0 : Math.min(final, Math.max(0, v));
    }

    return out;
  }, [confusionFinal, reducedMotion, totalCells, cellTarget, t, effectiveClasses]);

  const metricsVisible = useMemo(() => {
    const local = reducedMotion ? 1 : clamp01((p - 0.45) / 0.35);
    return metricsFinal.map((m) => ({ ...m, shown: clamp01(m.value * local) }));
  }, [metricsFinal, p, reducedMotion]);

  const confusionMax = useMemo(() => {
    const flat = confusionFinal.flat();
    return flat.length ? Math.max(...flat) : 1;
  }, [confusionFinal]);

  const curveVisible = useMemo(() => {
    const prSorted = [...prCurveFull].sort((a, b) => a.x - b.x);
    const rocSorted = [...rocCurveFull].sort((a, b) => a.x - b.x);
    if (reducedMotion) return { pr: prSorted, roc: rocSorted };
    const local = clamp01((p - 0.4) / 0.35);
    const prN = Math.max(2, Math.floor(lerp(2, prSorted.length, local)));
    const rocN = Math.max(2, Math.floor(lerp(2, rocSorted.length, local)));
    return { pr: prSorted.slice(0, prN), roc: rocSorted.slice(0, rocN) };
  }, [prCurveFull, rocCurveFull, p, reducedMotion]);

  // Artifact writing near completion.
  const wroteRef = useRef({ confusion: false, metrics: false });
  useEffect(() => {
    if (!writeArtifact) return;
    if (reducedMotion || p > 0.86) {
      if (!wroteRef.current.confusion) {
        wroteRef.current.confusion = true;
        writeArtifact('/artifacts/confusion_matrix.json', {
          classes: classCount,
          matrix: confusionFinal,
        });
      }
      if (!wroteRef.current.metrics) {
        wroteRef.current.metrics = true;
        writeArtifact('/artifacts/metrics.json', {
          classes: effectiveClasses,
          metrics: Object.fromEntries(metricsFinal.map((m) => [m.key, m.value])),
        });
      }
    }
  }, [confusionFinal, metricsFinal, p, reducedMotion, writeArtifact, effectiveClasses]);

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-replit-border/60 bg-replit-surface/35 backdrop-blur p-6">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold text-replit-text">Model Evaluation Dashboard</div>
            <div className="text-sm text-replit-textMuted">Confusion Matrix, PR/ROC Curves, and Performance Metrics</div>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-6 gap-3 mb-6">
          {metricsVisible.map((m) => (
            <div key={m.key} className="rounded-lg border border-replit-border/60 bg-replit-surface/35 p-3">
              <div className="text-[11px] text-replit-textMuted">{m.label}</div>
              <div className="text-lg font-mono text-replit-text">{m.shown.toFixed(3)}</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          <div className="rounded-xl border border-replit-border/60 bg-replit-bg p-4">
            <div className="text-sm text-replit-textMuted mb-3">Precision-Recall Curve</div>
            <div className="h-[280px] rounded-lg border border-replit-border/60 bg-replit-surface/20 overflow-hidden">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={curveVisible.pr.map((pt) => ({ recall: pt.x, precision: pt.y }))}
                  margin={{ top: 8, right: 16, bottom: 12, left: 12 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="recall"
                    stroke="#9CA3AF"
                    domain={[0, 1]}
                    ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    label={{ value: 'Recall', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
                  />
                  <YAxis
                    dataKey="precision"
                    stroke="#9CA3AF"
                    domain={[0, 1]}
                    ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    label={{ value: 'Precision', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  />
                  <Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={3} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-xl border border-replit-border/60 bg-replit-bg p-4">
            <div className="text-sm text-replit-textMuted mb-3">ROC Curve</div>
            <div className="h-[280px] rounded-lg border border-replit-border/60 bg-replit-surface/20 overflow-hidden">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={curveVisible.roc.map((pt) => ({ fpr: pt.x, tpr: pt.y }))}
                  margin={{ top: 8, right: 16, bottom: 12, left: 12 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="fpr"
                    stroke="#9CA3AF"
                    domain={[0, 1]}
                    ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }}
                  />
                  <YAxis
                    dataKey="tpr"
                    stroke="#9CA3AF"
                    domain={[0, 1]}
                    ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                    tickFormatter={(v: number) => v.toFixed(1)}
                    label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  />
                  <Line type="monotone" dataKey="tpr" stroke="#f59e0b" strokeWidth={3} dot={false} isAnimationActive={false} />
                  <Line
                    type="monotone"
                    dataKey="tpr"
                    data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]}
                    stroke="#64748b"
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {showMatrix ? (
          <div className="rounded-xl border border-replit-border/60 bg-replit-bg p-6 overflow-hidden">
            <div className="text-sm text-replit-textMuted mb-4">Confusion Matrix</div>
            <div className="flex items-center justify-center overflow-hidden">
              <div className="inline-block rounded-lg p-4 border border-replit-border/60 bg-replit-surface/30 max-h-[520px] overflow-auto">
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
                                  backgroundColor: `rgba(99,102,241,${Math.min(0.9, 0.2 + (cell / Math.max(1, confusionMax)) * 0.7)})`,
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
        ) : null}
      </div>
    </div>
  );
}
