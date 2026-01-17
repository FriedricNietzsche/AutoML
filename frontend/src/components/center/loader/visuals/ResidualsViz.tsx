import { useMemo } from 'react';
import { CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import type { VisualProps } from '../types';
import { clamp01 } from '../types';

type ResidualPoint = { id: number; predicted: number; actual: number; error: number };

function getPointColor(error: number) {
  const absError = Math.abs(error);
  if (absError < 3) return '#10b981';
  if (absError < 6) return '#f59e0b';
  return '#ef4444';
}

export default function ResidualsViz({ reducedMotion, residuals, phaseProgress }: VisualProps) {
  const data = useMemo<ResidualPoint[]>(() => {
    return (residuals ?? []).map((p, idx) => ({
      id: idx,
      predicted: p.pred,
      actual: p.true,
      error: p.residual,
    }));
  }, [residuals]);

  const revealProgress = reducedMotion ? 1 : clamp01(phaseProgress);
  const visibleCount = Math.max(0, Math.floor(data.length * revealProgress));
  const visibleData = useMemo(() => data.slice(0, visibleCount), [data, visibleCount]);

  const metrics = useMemo(() => {
    if (visibleData.length === 0) {
      return { meanError: 0, mse: 0, rmse: 0, mae: 0, r2: 0 };
    }
    const errors = visibleData.map((p) => p.error);
    const meanError = errors.reduce((a, b) => a + b, 0) / errors.length;
    const mse = errors.reduce((a, e) => a + e * e, 0) / errors.length;
    const rmse = Math.sqrt(mse);
    const mae = errors.reduce((a, e) => a + Math.abs(e), 0) / errors.length;
    const actualValues = visibleData.map((p) => p.actual);
    const meanActual = actualValues.reduce((a, b) => a + b, 0) / actualValues.length;
    const ssTot = actualValues.reduce((a, v) => a + Math.pow(v - meanActual, 2), 0);
    const ssRes = errors.reduce((a, e) => a + e * e, 0);
    const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
    return { meanError, mse, rmse, mae, r2 };
  }, [visibleData]);

  const domainStep = (min: number, max: number) => {
    const span = Math.max(1e-6, max - min);
    const rough = span / 5;
    const pow10 = Math.pow(10, Math.floor(Math.log10(rough)));
    const x = rough / pow10;
    const base = x <= 1 ? 1 : x <= 2 ? 2 : x <= 5 ? 5 : 10;
    return base * pow10;
  };

  const { xDomain, yDomain } = useMemo(() => {
    if (visibleData.length === 0) {
      return { xDomain: [0, 100] as [number, number], yDomain: [-10, 10] as [number, number] };
    }
    const xs = visibleData.map((p) => p.predicted);
    const ys = visibleData.map((p) => p.error);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xStep = domainStep(xMin, xMax);
    const yStep = domainStep(yMin, yMax);
    return {
      xDomain: [Math.floor(xMin / xStep) * xStep, Math.ceil(xMax / xStep) * xStep] as [number, number],
      yDomain: [Math.floor(yMin / yStep) * yStep, Math.ceil(yMax / yStep) * yStep] as [number, number],
    };
  }, [visibleData]);

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-replit-border/60 bg-replit-surface/35 backdrop-blur p-6 pb-5 h-full flex flex-col min-h-0">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold text-replit-text">Residuals Plot</div>
            <div className="text-sm text-replit-textMuted">Predicted vs Error — Regression Analysis</div>
          </div>
          <div className="text-xs font-mono text-replit-textMuted">n={visibleData.length}</div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-6">
          <div className="bg-replit-accent/10 border border-replit-accent/30 rounded-lg p-3">
            <div className="text-replit-accent text-xs font-semibold mb-1">Mean Error</div>
            <div className="text-replit-text text-lg font-mono">{metrics.meanError.toFixed(3)}</div>
          </div>
          <div className="bg-replit-warning/10 border border-replit-warning/30 rounded-lg p-3">
            <div className="text-replit-warning text-xs font-semibold mb-1">RMSE</div>
            <div className="text-replit-text text-lg font-mono">{metrics.rmse.toFixed(3)}</div>
          </div>
          <div className="bg-replit-success/10 border border-replit-success/30 rounded-lg p-3">
            <div className="text-replit-success text-xs font-semibold mb-1">MAE</div>
            <div className="text-replit-text text-lg font-mono">{metrics.mae.toFixed(3)}</div>
          </div>
          <div className="bg-replit-surface/40 border border-replit-border/60 rounded-lg p-3">
            <div className="text-replit-textMuted text-xs font-semibold mb-1">MSE</div>
            <div className="text-replit-text text-lg font-mono">{metrics.mse.toFixed(3)}</div>
          </div>
          <div className="bg-replit-info/10 border border-replit-info/30 rounded-lg p-3">
            <div className="text-replit-info text-xs font-semibold mb-1">R² Score</div>
            <div className="text-replit-text text-lg font-mono">{metrics.r2.toFixed(3)}</div>
          </div>
        </div>

        <div className="bg-replit-bg/30 rounded-xl p-4 mb-6 relative overflow-hidden border border-replit-border/60 flex-1 min-h-0">
          {data.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center bg-replit-bg/40 backdrop-blur-sm z-10 rounded-xl">
              <div className="text-center">
                <div className="w-10 h-10 border-4 border-replit-accent border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                <p className="text-replit-textMuted">Waiting for residual data…</p>
              </div>
            </div>
          )}
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.25)" />
              <XAxis
                type="number"
                dataKey="predicted"
                stroke="rgb(var(--replit-text-muted))"
                domain={xDomain}
                label={{ value: 'Predicted Value', position: 'insideBottom', offset: -10, fill: 'rgb(var(--replit-text-muted))' }}
              />
              <YAxis
                type="number"
                dataKey="error"
                stroke="rgb(var(--replit-text-muted))"
                domain={yDomain}
                label={{ value: 'Residual (Actual - Predicted)', angle: -90, position: 'insideLeft', fill: 'rgb(var(--replit-text-muted))' }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: 'rgba(20,24,36,0.9)', border: '1px solid rgba(148,163,184,0.25)', borderRadius: '8px' }}
                formatter={(v, name) => [Number(v).toFixed(3), name]}
              />
              <ReferenceLine y={0} stroke="rgb(var(--replit-text-muted))" strokeWidth={2} strokeDasharray="5 5" />
              <Scatter name="Residuals" data={visibleData} isAnimationActive={false}>
                {visibleData.map((entry) => (
                  <Cell key={entry.id} fill={getPointColor(entry.error)} opacity={0.6} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#10b981' }}></div>
            <span className="text-replit-textMuted text-xs">Good (|error| &lt; 3)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#f59e0b' }}></div>
            <span className="text-replit-textMuted text-xs">Moderate (3 ≤ |error| &lt; 6)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#ef4444' }}></div>
            <span className="text-replit-textMuted text-xs">Poor (|error| ≥ 6)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
