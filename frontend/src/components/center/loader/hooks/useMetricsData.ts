import { useMemo, useState, useEffect } from 'react';
import { buildLossSeries, buildAccuracySeries, type LossPoint, type MetricPoint } from '../utils/loaderHelpers';
import { clamp01 } from '../types';

interface MetricsState {
  lossSeries: LossPoint[];
  accSeries: MetricPoint[];
  f1Series: MetricPoint[];
  rmseSeries: MetricPoint[];
}

interface UseMetricsDataProps {
  metricsState: MetricsState;
  useMockStream: boolean;
  metricKind: 'accuracy' | 'f1' | 'rmse';
  phase: { kind: string; graphType?: string };
  phaseProgress: number;
}

/**
 * Custom hook that manages metrics data visibility and progressive reveal
 */
export function useMetricsData({
  metricsState,
  useMockStream,
  metricKind,
  phase,
  phaseProgress,
}: UseMetricsDataProps) {
  const [lossVisible, setLossVisible] = useState<LossPoint[]>([]);
  const [accVisible, setAccVisible] = useState<MetricPoint[]>([]);

  // Build full series (fallback if mock data not available)
  const lossFull = useMemo(() => {
    if (useMockStream && metricsState.lossSeries.length > 0) return metricsState.lossSeries;
    return buildLossSeries(36);
  }, [metricsState.lossSeries, useMockStream]);

  const accFull = useMemo(() => {
    if (useMockStream && metricsState.accSeries.length > 0) return metricsState.accSeries;
    return buildAccuracySeries(36);
  }, [metricsState.accSeries, useMockStream]);

  const f1Full = useMemo(() => {
    if (useMockStream && metricsState.f1Series.length > 0) return metricsState.f1Series;
    return [] as MetricPoint[];
  }, [metricsState.f1Series, useMockStream]);

  const rmseFull = useMemo(() => {
    if (useMockStream && metricsState.rmseSeries.length > 0) return metricsState.rmseSeries;
    return [] as MetricPoint[];
  }, [metricsState.rmseSeries, useMockStream]);

  const metricFull = useMemo(() => {
    if (metricKind === 'rmse') return rmseFull;
    if (metricKind === 'f1') return f1Full.length > 0 ? f1Full : accFull;
    return accFull;
  }, [accFull, f1Full, metricKind, rmseFull]);

  // Progressive reveal based on phase progress
  useEffect(() => {
    if (phase.kind !== 'graph') return;
    const graphType = phase.graphType;

    const late = clamp01((phaseProgress - 0.12) / 0.88);

    if (graphType === 'loss') {
      const targetCount = Math.max(0, Math.floor(lossFull.length * late));
      if (targetCount > lossVisible.length) {
        const next = lossFull.slice(0, targetCount);
        setLossVisible(next);
      }
    }

    if (graphType === 'accuracy') {
      const series = metricFull;
      const targetCount = Math.max(0, Math.floor(series.length * late));
      if (targetCount > accVisible.length) {
        const next = series.slice(0, targetCount);
        setAccVisible(next);
      }
    }
  }, [phase, phaseProgress, lossFull, metricFull, lossVisible.length, accVisible.length]);

  return {
    lossVisible,
    accVisible,
    lossFull,
    accFull,
    metricFull,
    setLossVisible,
    setAccVisible,
  };
}
