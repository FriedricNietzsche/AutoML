import type { ReactElement } from 'react';
import { useMemo } from 'react';
import type { VisualBaseProps } from '../types';
import { clamp01, lerp, seeded } from '../types';

type NeuralNetVizProps = VisualBaseProps & {
  mode: 'forward' | 'backprop';
};

type Pulse = {
  id: string;
  fromLayer: number;
  toLayer: number;
  fromNeuron: number;
  toNeuron: number;
  progress: number; // 0..1
};

function lerpPt(a: { x: number; y: number }, b: { x: number; y: number }, t: number) {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) };
}

function easeInOut(t: number) {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

function getNeuronPosition({
  layerIdx,
  neuronIdx,
  totalNeurons,
  width,
  height,
  padX,
  layerCount,
}: {
  layerIdx: number;
  neuronIdx: number;
  totalNeurons: number;
  width: number;
  height: number;
  padX: number;
  layerCount: number;
}) {
  const x = padX + (layerIdx * (width - padX * 2)) / Math.max(1, layerCount - 1);
  const centerY = height / 2;
  const spread = 54;
  const y = centerY + (neuronIdx - (totalNeurons - 1) / 2) * spread;
  return { x, y };
}

export default function NeuralNetViz({ timeMs, phaseProgress, seed, mode, reducedMotion }: NeuralNetVizProps) {
  const t = reducedMotion ? 0 : timeMs / 1000;
  const p = reducedMotion ? 1 : clamp01(phaseProgress);

  const width = 800;
  const height = 400;
  const padX = 90;
  const neuronRadius = 16;
  const layers = useMemo(() => [4, 6, 6, 4] as const, []);
  const layerCount = layers.length;

  // Active layer sweeps through; forward: 0→end, backprop: end→0.
  const activeLayer = useMemo(() => {
    if (reducedMotion) return mode === 'forward' ? layerCount - 1 : 0;
    const idx = Math.min(layerCount - 1, Math.floor(p * layerCount));
    return mode === 'forward' ? idx : (layerCount - 1 - idx);
  }, [layerCount, mode, p, reducedMotion]);

  // Only propagate between one layer-pair at a time.
  // Forward: activeLayer -> activeLayer+1
  // Backprop: activeLayer -> activeLayer-1
  const activeFromLayer = useMemo(() => {
    if (mode === 'forward') return Math.min(layerCount - 2, activeLayer);
    return Math.max(1, activeLayer);
  }, [activeLayer, layerCount, mode]);

  const activeToLayer = mode === 'forward' ? activeFromLayer + 1 : activeFromLayer - 1;
  // Connection lines are defined between layer k -> k+1, so map backprop to that edge index.
  const activeEdgeLayer = mode === 'forward' ? activeFromLayer : activeToLayer;

  const colors = {
    base: 'rgb(var(--replit-border))',
    dim: 'rgb(var(--replit-text-muted))',
    forward: 'rgb(var(--replit-accent))',
    forwardPulse: 'rgb(var(--replit-accent-hover))',
    backward: 'rgb(var(--replit-warning))',
    backwardPulse: 'rgb(var(--replit-warning))',
  };

  const primary = mode === 'forward' ? colors.forward : colors.backward;
  const pulseColor = mode === 'forward' ? colors.forwardPulse : colors.backwardPulse;

  const pulses = useMemo((): Pulse[] => {
    if (reducedMotion) return [];
    const pulses: Pulse[] = [];

    // Deterministic pulse emission using time buckets.
    const tickMs = 130;
    const pulseDurationMs = 760;
    const now = timeMs;
    const currentTick = Math.floor(now / tickMs);

    // Look back enough ticks to cover the duration.
    const lookbackTicks = Math.ceil(pulseDurationMs / tickMs) + 1;

    const localSeed = seed * 911 + 1337;

    for (let tick = currentTick - lookbackTicks; tick <= currentTick; tick += 1) {
      const tickStart = tick * tickMs;
      const age = now - tickStart;
      if (age < 0 || age > pulseDurationMs) continue;
      const prog = clamp01(age / pulseDurationMs);

      // Emit a couple of pulses per tick, but ONLY for the active layer transition.
      const fromLayer = activeFromLayer;
      const toLayer = activeToLayer;

      // Higher probability so the active edge feels alive.
      for (let slot = 0; slot < 2; slot += 1) {
        const r = seeded(localSeed + tick * 97 + fromLayer * 313 + slot * 911);
        if (r >= 0.8) continue;

        const fromNeuron = Math.floor(seeded(localSeed + tick * 131 + fromLayer * 503 + slot * 97 + 11) * layers[fromLayer]);
        const toNeuron = Math.floor(seeded(localSeed + tick * 151 + toLayer * 547 + slot * 101 + 29) * layers[toLayer]);

        pulses.push({
          id: `t${tick}-L${fromLayer}-S${slot}`,
          fromLayer,
          toLayer,
          fromNeuron,
          toNeuron,
          progress: prog,
        });
      }
    }

    return pulses;
  }, [activeFromLayer, activeToLayer, layers, reducedMotion, seed, timeMs]);

  const connections = useMemo(() => {
    const lines: Array<{ key: string; x1: number; y1: number; x2: number; y2: number; layer: number }> = [];
    for (let layer = 0; layer < layerCount - 1; layer += 1) {
      for (let j = 0; j < layers[layer]; j += 1) {
        for (let k = 0; k < layers[layer + 1]; k += 1) {
          const from = getNeuronPosition({
            layerIdx: layer,
            neuronIdx: j,
            totalNeurons: layers[layer],
            width,
            height,
            padX,
            layerCount,
          });
          const to = getNeuronPosition({
            layerIdx: layer + 1,
            neuronIdx: k,
            totalNeurons: layers[layer + 1],
            width,
            height,
            padX,
            layerCount,
          });
          lines.push({
            key: `${layer}-${j}-${k}`,
            x1: from.x,
            y1: from.y,
            x2: to.x,
            y2: to.y,
            layer,
          });
        }
      }
    }
    return lines;
  }, [height, layerCount, layers, padX, width]);

  return (
    <div className="w-full">
      <div className="rounded-2xl border border-replit-border/60 bg-replit-surface/35 backdrop-blur p-6">
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm text-replit-textMuted">
            Neural Network · <span className="text-replit-text">{mode === 'forward' ? 'Forward Pass' : 'Backprop'}</span>
          </div>
          <div className="text-xs font-mono text-replit-textMuted">layers={layerCount} · active={activeLayer + 1}</div>
        </div>

        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-[430px]" role="img" aria-label="Neural network propagation">
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Edges */}
          {connections.map((c) => {
            const isActiveEdge = c.layer === activeEdgeLayer;

            const opacity = isActiveEdge ? 0.22 : 0.05;

            // Only animate flow on the active connection layer.
            const dash = !reducedMotion && isActiveEdge ? '8 14' : undefined;
            const dashOffset = !reducedMotion && isActiveEdge ? -((t * 46) % 100) : undefined;

            return (
              <line
                key={c.key}
                x1={c.x1}
                y1={c.y1}
                x2={c.x2}
                y2={c.y2}
                stroke={primary}
                strokeOpacity={opacity}
                strokeWidth={1.6}
                strokeLinecap="round"
                strokeDasharray={dash}
                strokeDashoffset={dashOffset}
              />
            );
          })}

          {/* Pulses */}
          {pulses.map((pulse) => {
            const from = getNeuronPosition({
              layerIdx: pulse.fromLayer,
              neuronIdx: pulse.fromNeuron,
              totalNeurons: layers[pulse.fromLayer],
              width,
              height,
              padX,
              layerCount,
            });
            const to = getNeuronPosition({
              layerIdx: pulse.toLayer,
              neuronIdx: pulse.toNeuron,
              totalNeurons: layers[pulse.toLayer],
              width,
              height,
              padX,
              layerCount,
            });

            const prog = easeInOut(pulse.progress);
            const pt = lerpPt(from, to, prog);
            const opacity = Math.sin(prog * Math.PI) * 0.9;
            const size = 3 + Math.sin(prog * Math.PI) * 2.4;

            return (
              <g key={pulse.id}>
                <circle cx={pt.x} cy={pt.y} r={size + 6} fill={pulseColor} opacity={opacity * 0.18} />
                <circle cx={pt.x} cy={pt.y} r={size + 10} fill={pulseColor} opacity={opacity * 0.07} />
                <circle cx={pt.x} cy={pt.y} r={size} fill={pulseColor} opacity={opacity} filter="url(#glow)" />
              </g>
            );
          })}

          {/* Nodes */}
          {layers.flatMap((count, layerIdx) => {
            const isActiveLayer = layerIdx === activeLayer;
            const intensity = reducedMotion ? 0.75 : isActiveLayer ? 0.7 + Math.sin(t * 2.0) * 0.3 : 0.28;
            const fill =
              mode === 'forward'
                ? layerIdx <= activeLayer
                  ? `rgb(var(--replit-accent) / ${clamp01(intensity)})`
                  : 'rgb(var(--replit-surface))'
                : layerIdx >= activeLayer
                  ? `rgb(var(--replit-warning) / ${clamp01(intensity)})`
                  : 'rgb(var(--replit-surface))';

            const stroke = mode === 'forward' ? colors.forward : colors.backward;

            const nodes: ReactElement[] = [];
            for (let i = 0; i < count; i += 1) {
              const pos = getNeuronPosition({ layerIdx, neuronIdx: i, totalNeurons: count, width, height, padX, layerCount });
              const haloR = neuronRadius + 10 + (reducedMotion ? 0 : 4 * Math.sin(t * 2.2 + i));
              const innerOpacity = reducedMotion ? 0.6 : 0.45 + 0.25 * Math.sin(t * 2.6 + i);

              nodes.push(
                <g key={`${layerIdx}-${i}`}>
                  {isActiveLayer && !reducedMotion ? (
                    <circle cx={pos.x} cy={pos.y} r={haloR} fill={stroke} opacity={0.16} />
                  ) : null}
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r={neuronRadius}
                    fill={fill}
                    stroke={stroke}
                    strokeOpacity={isActiveLayer ? 0.95 : 0.5}
                    strokeWidth={2}
                    filter={isActiveLayer ? 'url(#glow)' : undefined}
                  />
                  {isActiveLayer ? (
                    <circle cx={pos.x} cy={pos.y} r={neuronRadius - 6} fill={stroke} opacity={innerOpacity} />
                  ) : null}
                </g>
              );
            }
            return nodes;
          })}

          {/* Labels */}
          <text x={40} y={height - 18} fill={colors.dim} fontSize={12} fontWeight={600}>
            INPUT
          </text>
          <text x={width - 110} y={height - 18} fill={colors.dim} fontSize={12} fontWeight={600}>
            OUTPUT
          </text>
        </svg>

        <div className="mt-4 text-xs text-replit-textMuted">
          {reducedMotion
            ? 'Reduced motion: static snapshot'
            : mode === 'forward'
              ? '→ Computing activations layer by layer'
              : '← Propagating gradients and updating weights'}
        </div>
      </div>
    </div>
  );
}
