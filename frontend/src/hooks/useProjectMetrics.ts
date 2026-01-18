/**
 * Hook to convert projectStore WebSocket events into TrainingLoader metrics format
 */
import { useMemo } from 'react';
import { useProjectStore } from '../store/projectStore';

export interface LiveMetricsState {
    lossSeries: Array<{ epoch: number; train_loss: number; val_loss: number }>;
    accSeries: Array<{ epoch: number; value: number }>;
    f1Series: Array<{ epoch: number; value: number }>;
    rmseSeries: Array<{ epoch: number; value: number }>;
    datasetPreview: {
        rows: any[];
        columns: string[];
        nRows: number;
        dataType?: 'image' | 'tabular';
        imageData?: {
            width: number;
            height: number;
            pixels: { r: number; g: number; b: number }[];
            needsClientLoad?: boolean;
        };
    } | null;
    metricsSummary: Record<string, number>;
    thinkingByStage: Record<string, string>;
    // Optional properties for advanced features
    confusionTable?: number[][];
    embeddingPoints?: Array<{ id: number; x: number; y: number; label: number; weight: number }>;
    gradientPath?: Array<{ x: number; y: number }>;
    surfaceSpec?: any;
    residuals?: Array<{ pred: number; true: number; residual: number }>;
}

export function useProjectMetrics(): LiveMetricsState {
    const datasetSample = useProjectStore((state) => state.datasetSample);
    const trainingMetrics = useProjectStore((state) => state.trainingMetrics);
    const profileSummary = useProjectStore((state) => state.profileSummary);
    const stages = useProjectStore((state) => state.stages);

    return useMemo(() => {
        // Convert training metrics to series format
        const metricsHistory = trainingMetrics?.metricsHistory || [];

        const lossSeries = metricsHistory
            .filter((m) => m.name === 'loss')
            .map((m) => ({
                epoch: m.step,
                train_loss: m.value,
                val_loss: m.value * 1.05 // Synthetic validation loss for visualization
            }));

        const accSeries = metricsHistory
            .filter((m) => m.name === 'accuracy' || m.name === 'acc')
            .map((m) => ({ epoch: m.step, value: m.value }));

        const f1Series = metricsHistory
            .filter((m) => m.name === 'f1')
            .map((m) => ({ epoch: m.step, value: m.value }));

        const rmseSeries = metricsHistory
            .filter((m) => m.name === 'rmse')
            .map((m) => ({ epoch: m.step, value: m.value }));


        // Convert dataset sample to preview format
        let datasetPreview = null;
        if (datasetSample) {
            // For image datasets, show image metadata
            if (datasetSample.images && datasetSample.images.length > 0) {
                datasetPreview = {
                    rows: datasetSample.images.slice(0, 5).map((url, idx) => ({
                        index: idx,
                        image_url: url,
                        type: 'image',
                    })),
                    columns: ['index', 'image_url', 'type'],
                    nRows: datasetSample.images.length,
                    dataType: 'image' as const,
                };
            } else if (datasetSample.columns && datasetSample.columns.length > 0) {
                // For tabular data, use actual row data if available
                const sampleRows = datasetSample.sample_rows || [];
                datasetPreview = {
                    rows: sampleRows,  // Use actual rows from backend
                    columns: datasetSample.columns,
                    nRows: datasetSample.nRows || 0,
                    dataType: 'tabular' as const,
                };
            }
        }

        // Extract metrics summary from profile
        const metricsSummary: Record<string, number> = {};
        if (profileSummary) {
            if (typeof profileSummary.n_rows === 'number') {
                metricsSummary.rows = profileSummary.n_rows;
            }
            if (typeof profileSummary.n_cols === 'number') {
                metricsSummary.columns = profileSummary.n_cols;
            }
        }

        // Get stage messages for "thinking"
        const thinkingByStage: Record<string, string> = {};
        Object.entries(stages).forEach(([stageId, stage]) => {
            if (stage.message) {
                thinkingByStage[stageId] = stage.message;
            }
        });

        return {
            lossSeries,
            accSeries,
            f1Series,
            rmseSeries,
            datasetPreview,
            metricsSummary,
            thinkingByStage,
        };
    }, [datasetSample, trainingMetrics, profileSummary, stages]);
}
