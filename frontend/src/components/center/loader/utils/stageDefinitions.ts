import type { StepDef } from '../types';
import type { LoaderStepId } from '../../../../mock/scenarioVizConfig';

/**
 * Fixed navigation nodes shown at the bottom of the loader
 */
export const FIXED_NODES = [
  { id: 1, label: 'Data Load' },
  { id: 2, label: 'Preprocess' },
  { id: 3, label: 'Train' },
  { id: 4, label: 'Evaluate' },
  { id: 5, label: 'Export' },
];

/**
 * Stage definitions for the training loader pipeline
 */
export const STAGE_DEFINITIONS: StepDef[] = [
  {
    id: 'matrixOps' as LoaderStepId,
    title: 'Loading Data',
    subtitle: 'Reading and validating dataset',
    durationMs: 6000,
    phases: [
      { kind: 'operation' as const },
      // Optional lightweight view of samples/embeddings if backend sends them
      { kind: 'visual' as const, visualId: 'embeddingScatter' as const, weight: 0.6 },
    ],
  },
  {
    id: 'preprocessing' as LoaderStepId,
    title: 'Preprocessing',
    subtitle: 'Normalizing and transforming features',
    durationMs: 6000,
    matrixLabel: 'Data Preview',
    matrixRows: 8,
    matrixCols: 6,
    phases: [
      { kind: 'operation' as const },
      // Residuals view doubles as a distribution/quality check when available
      { kind: 'visual' as const, visualId: 'residuals' as const, weight: 0.5 },
    ],
  },
  {
    id: 'trainLoss' as LoaderStepId,
    title: 'Training Model',
    subtitle: 'Optimizing model parameters',
    durationMs: 8000,
    equations: ['\\mathcal{L}(\\theta)'],
    phases: [
      { kind: 'graph' as const, graphType: 'loss' as const, weight: 0.6 },
      { kind: 'visual' as const, visualId: 'gradDescent' as const, weight: 0.4 },
    ],
  },
  {
    id: 'evaluation' as LoaderStepId,
    title: 'Evaluating Performance',
    subtitle: 'Computing metrics and validation',
    durationMs: 7000,
    equations: ['\\mathrm{F1}=2\\frac{PR}{P+R}'],
    phases: [
      { kind: 'visual' as const, visualId: 'evaluation' as const, weight: 0.4 },
      { kind: 'visual' as const, visualId: 'confusionMatrix' as const, weight: 0.3 },
      { kind: 'visual' as const, visualId: 'residuals' as const, weight: 0.3 },
    ],
  },
  {
    id: 'embedding' as LoaderStepId,
    title: 'Exporting Model',
    subtitle: 'Packaging for deployment',
    durationMs: 5000,
    equations: ['\\mathbf{z}=f(\\mathbf{x})'],
    phases: [
      { kind: 'visual' as const, visualId: 'embeddingScatter' as const, weight: 0.6 },
      { kind: 'visual' as const, visualId: 'evaluation' as const, weight: 0.4 },
    ],
  },
];

/**
 * Get stage-specific prompt messages
 */
export function getStagePrompt(
  currentStage: number,
  isStageRunning: boolean,
  stageCompleted: boolean,
  stepTitle: string,
  stepSubtitle: string
) {
  if (currentStage === 0) {
    return {
      title: "Ready to Build Your AI Model?",
      subtitle: "Click 'Proceed' to start the automated machine learning pipeline. We'll guide you through data loading, preprocessing, training, evaluation, and deployment.",
    };
  }
  if (isStageRunning) {
    return {
      title: stepTitle,
      subtitle: stepSubtitle,
    };
  }
  if (stageCompleted) {
    const stageMessages = [
      { title: "Stage 1 Complete!", subtitle: "Data has been loaded and validated. Ready to proceed to preprocessing?" },
      { title: "Stage 2 Complete!", subtitle: "Data preprocessing finished successfully. Ready to train the model?" },
      { title: "Stage 3 Complete!", subtitle: "Model training completed. Ready to evaluate performance?" },
      { title: "Stage 4 Complete!", subtitle: "Model evaluation finished. Review the results and decide next steps." },
      { title: "Deployment Complete!", subtitle: "Redirecting to model tester..." },
    ];
    return stageMessages[currentStage - 1] || stageMessages[0];
  }
  return { title: "", subtitle: "" };
}
