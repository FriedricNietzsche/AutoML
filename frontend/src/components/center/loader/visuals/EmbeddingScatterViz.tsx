import { useMemo } from 'react';
import { Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts';
import type { VisualProps } from '../types';
import { clamp01 } from '../types';

type ScatterPoint = { id: number; x: number; y: number; cluster: number; epoch: number };

function generateClusterColor(clusterIdx: number, totalClusters: number) {
  const hue = (clusterIdx * 360) / Math.max(totalClusters, 20);
  const saturation = 65 + (clusterIdx % 3) * 10;
  const lightness = 55 + (clusterIdx % 2) * 5;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

export default function EmbeddingScatterViz({ points, phaseProgress, reducedMotion }: VisualProps) {
  const data = useMemo<ScatterPoint[]>(() => {
    if (!points || points.length === 0) return [];
    const sorted = [...points].sort((a, b) => a.id - b.id);
    const pointsPerEpoch = 5;
    return sorted.map((pt, idx) => ({
      id: pt.id,
      x: pt.x,
      y: pt.y,
      cluster: pt.label,
      epoch: Math.floor(idx / pointsPerEpoch),
    }));
  }, [points]);

  const maxEpoch = data.length ? Math.max(...data.map((p) => p.epoch)) : 0;
  const revealProgress = reducedMotion ? 1 : clamp01(phaseProgress);
  const currentEpoch = Math.max(0, Math.floor(maxEpoch * revealProgress));
  const visibleData = useMemo(() => data.filter((p) => p.epoch <= currentEpoch), [data, currentEpoch]);

  const allClusterKeys = useMemo(() => {
    const keys = new Set<number>();
    for (const p of data) keys.add(p.cluster);
    return Array.from(keys).sort((a, b) => a - b);
  }, [data]);
  const totalClusters = allClusterKeys.length || 1;

  const stats = useMemo(() => {
    const clusterCounts = new Map<number, number>();
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of visibleData) {
      clusterCounts.set(p.cluster, (clusterCounts.get(p.cluster) ?? 0) + 1);
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    return { clusterCounts, minX, maxX, minY, maxY };
  }, [visibleData]);

  const clusterKeys = Array.from(stats.clusterCounts.keys()).sort((a, b) => a - b);
  const clustersCount = clusterKeys.length || totalClusters;

  const xPadding = stats.minX !== Infinity ? Math.max(0.5, (stats.maxX - stats.minX) * 0.15) : 1;
  const yPadding = stats.minY !== Infinity ? Math.max(0.5, (stats.maxY - stats.minY) * 0.15) : 1;
  const rawXMin = stats.minX !== Infinity ? stats.minX - xPadding : -10;
  const rawXMax = stats.maxX !== Infinity ? stats.maxX + xPadding : 10;
  const rawYMin = stats.minY !== Infinity ? stats.minY - yPadding : -10;
  const rawYMax = stats.maxY !== Infinity ? stats.maxY + yPadding : 10;

  const domainStep = (min: number, max: number) => {
    const span = Math.max(1e-6, max - min);
    const rough = span / 5;
    const pow10 = Math.pow(10, Math.floor(Math.log10(rough)));
    const x = rough / pow10;
    const base = x <= 1 ? 1 : x <= 2 ? 2 : x <= 5 ? 5 : 10;
    return base * pow10;
  };

  const xStep = domainStep(rawXMin, rawXMax);
  const yStep = domainStep(rawYMin, rawYMax);
  const xDomain: [number, number] = [Math.floor(rawXMin / xStep) * xStep, Math.ceil(rawXMax / xStep) * xStep];
  const yDomain: [number, number] = [Math.floor(rawYMin / yStep) * yStep, Math.ceil(rawYMax / yStep) * yStep];

  const tooltipStyle = {
    backgroundColor: 'rgba(20,24,36,0.9)',
    border: '1px solid rgba(148,163,184,0.25)',
    borderRadius: '8px',
    color: '#fff',
  };

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-replit-border/60 bg-replit-surface/35 backdrop-blur p-6 pb-5">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold text-replit-text">Embedding Space Visualization</div>
            <div className="text-sm text-replit-textMuted">Point clusters converging over epochs</div>
          </div>
          <div className="text-xs font-mono text-replit-textMuted">n={visibleData.length}</div>
        </div>

        <div className="bg-replit-bg/30 rounded-xl p-4 mb-4 relative overflow-hidden border border-replit-border/60">
          {data.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center bg-replit-bg/40 backdrop-blur-sm z-10">
              <div className="text-center">
                <div className="w-10 h-10 border-4 border-replit-accent border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                <p className="text-replit-textMuted">Waiting for embedding data…</p>
              </div>
            </div>
          )}
          <ResponsiveContainer width="100%" height={460}>
            <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.25)" />
              <XAxis
                type="number"
                dataKey="x"
                stroke="rgb(var(--replit-text-muted))"
                domain={xDomain}
                tickFormatter={(v: number) => v.toFixed(2)}
                label={{
                  value: 'Embedding Dimension 1',
                  position: 'insideBottom',
                  offset: -10,
                  fill: 'rgb(var(--replit-text-muted))',
                }}
              />
              <YAxis
                type="number"
                dataKey="y"
                stroke="rgb(var(--replit-text-muted))"
                domain={yDomain}
                tickFormatter={(v: number) => v.toFixed(2)}
                label={{
                  value: 'Embedding Dimension 2',
                  angle: -90,
                  position: 'insideLeft',
                  fill: 'rgb(var(--replit-text-muted))',
                }}
              />
              <Tooltip
                cursor={{ stroke: 'rgb(var(--replit-border))', strokeOpacity: 0.5 }}
                contentStyle={tooltipStyle}
                formatter={(v: number | undefined, name?: string) => {
                  const value = Number(v ?? 0).toFixed(3);
                  if (name === 'x') return [value, 'X'];
                  if (name === 'y') return [value, 'Y'];
                  return [value, name ?? 'value'];
                }}
              />
              <Scatter name="Embeddings" data={visibleData} isAnimationActive={false}>
                {visibleData.map((entry) => (
                  <Cell
                    key={entry.id}
                    fill={generateClusterColor(entry.cluster, totalClusters)}
                    opacity={0.7}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center gap-3 mb-5 px-4 py-3 bg-replit-surface/40 rounded-lg border border-replit-border/60">
          <span className="text-replit-textMuted font-medium">Clusters:</span>
          <span className="text-replit-text font-mono text-lg">{clustersCount}</span>
          <span className="text-replit-textMuted mx-2">|</span>
          <span className="text-replit-textMuted font-medium">Epoch:</span>
          <span className="text-replit-text font-mono text-lg">{currentEpoch}</span>
          <span className="text-replit-textMuted mx-2">|</span>
          <span className="text-replit-textMuted font-medium">Points:</span>
          <span className="text-replit-text font-mono text-lg">{visibleData.length}</span>
        </div>

        <div className="grid grid-cols-6 gap-2 max-h-32 overflow-y-auto">
          {clusterKeys.map((cluster) => {
            const count = stats.clusterCounts.get(cluster) ?? 0;
            const color = generateClusterColor(cluster, totalClusters);
            return (
              <div
                key={cluster}
                className="rounded-lg p-2 border text-center"
                style={{ backgroundColor: `${color}20`, borderColor: `${color}50` }}
              >
                <div className="flex items-center justify-center gap-1 mb-1">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                  <span className="text-replit-text font-semibold text-xs">{cluster}</span>
                </div>
                <p className="text-replit-textMuted text-xs">{count}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
