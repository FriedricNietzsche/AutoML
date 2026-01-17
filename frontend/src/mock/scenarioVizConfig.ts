export type ScenarioId = 'A' | 'B' | 'C';

export type LoaderStepId =
  | 'neuralNet'
  | 'matrixOps'
  | 'gradientDescent'
  | 'trainLoss'
  | 'modelMetric'
  | 'embedding'
  | 'evaluation'
  | 'confusionMatrix'
  | 'residuals';

export type ScenarioVizConfig = {
  scenarioId: ScenarioId;
  scenarioName: string;
  modelLabel: string;
  steps: Array<{
    id: LoaderStepId;
    title: string;
    subtitle: string;
    enabled: boolean;
  }>;
  metricKind: 'accuracy' | 'f1' | 'rmse';
  showConfusionMatrix: boolean;
  showEmbedding: boolean;
  showResiduals: boolean;
};

export const SCENARIO_VIZ: Record<ScenarioId, ScenarioVizConfig> = {
  A: {
    scenarioId: 'A',
    scenarioName: 'Binary Classification (LogReg)',
    modelLabel: 'Logistic Regression',
    metricKind: 'accuracy',
    showConfusionMatrix: true,
    showEmbedding: true,
    showResiduals: false,
    steps: [
      { id: 'neuralNet', title: 'Neural Network', subtitle: 'Forward signal propagation', enabled: true },
      { id: 'matrixOps', title: 'Matrix Operations', subtitle: 'Batched multiply-accumulate', enabled: true },
      { id: 'gradientDescent', title: 'Gradient Descent', subtitle: 'Optimizing parameters', enabled: true },
      { id: 'trainLoss', title: 'Training Loss', subtitle: 'Train vs validation loss', enabled: true },
      { id: 'modelMetric', title: 'Model Metric', subtitle: 'Accuracy over epochs', enabled: true },
      { id: 'embedding', title: 'Embedding Space', subtitle: 'Cluster convergence', enabled: true },
      { id: 'evaluation', title: 'Evaluation', subtitle: 'Confusion matrix + PR curve', enabled: true },
      { id: 'residuals', title: 'Residuals', subtitle: 'Residual plot', enabled: false },
    ],
  },
  B: {
    scenarioId: 'B',
    scenarioName: 'Multiclass Classification (RandomForest)',
    modelLabel: 'Random Forest',
    metricKind: 'f1',
    showConfusionMatrix: true,
    showEmbedding: true,
    showResiduals: false,
    steps: [
      { id: 'neuralNet', title: 'Neural Network', subtitle: 'Forward signal propagation', enabled: true },
      { id: 'matrixOps', title: 'Matrix Operations', subtitle: 'Batched multiply-accumulate', enabled: true },
      { id: 'gradientDescent', title: 'Gradient Descent', subtitle: 'Optimizing parameters', enabled: true },
      { id: 'trainLoss', title: 'Training Loss', subtitle: 'Train vs validation loss', enabled: true },
      { id: 'modelMetric', title: 'Model Metric', subtitle: 'Macro-F1 over epochs', enabled: true },
      { id: 'embedding', title: 'Embedding Space', subtitle: 'Cluster convergence', enabled: true },
      { id: 'evaluation', title: 'Evaluation', subtitle: 'Confusion matrix + ROC curve', enabled: true },
      { id: 'residuals', title: 'Residuals', subtitle: 'Residual plot', enabled: false },
    ],
  },
  C: {
    scenarioId: 'C',
    scenarioName: 'Regression (Ridge)',
    modelLabel: 'Ridge Regression',
    metricKind: 'rmse',
    showConfusionMatrix: false,
    showEmbedding: false,
    showResiduals: true,
    steps: [
      { id: 'neuralNet', title: 'Neural Network', subtitle: 'Forward signal propagation', enabled: true },
      { id: 'matrixOps', title: 'Matrix Operations', subtitle: 'Batched multiply-accumulate', enabled: true },
      { id: 'gradientDescent', title: 'Gradient Descent', subtitle: 'Optimizing parameters', enabled: true },
      { id: 'trainLoss', title: 'Training Loss', subtitle: 'Train vs validation loss', enabled: true },
      { id: 'modelMetric', title: 'Model Metric', subtitle: 'RMSE over epochs', enabled: true },
      { id: 'embedding', title: 'Embedding Space', subtitle: 'Cluster convergence', enabled: false },
      { id: 'residuals', title: 'Residuals', subtitle: 'Residual vs prediction', enabled: true },
      { id: 'evaluation', title: 'Evaluation', subtitle: 'Regression summary', enabled: true },
    ],
  },
};
