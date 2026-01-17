import { clamp, createRng, type ScenarioBuilder, type ScenarioData } from './scenarioTypes';

export const scenarioC: ScenarioBuilder = (seed = 4096): ScenarioData => {
  const rng = createRng(seed + 33);
  const totalEpochs = 22;

  const lossCurve = Array.from({ length: totalEpochs }, (_, i) => {
    const epoch = i + 1;
    const base = 0.9 * Math.exp(-epoch / 7.5) + 0.18;
    const noise = (rng() - 0.5) * 0.03;
    const train_loss = clamp(base + noise, 0.1, 1.2);
    const val_loss = clamp(base * 1.12 + noise * 0.6 + 0.08, 0.15, 1.35);
    return { epoch, train_loss, val_loss };
  });

  const rmseCurve = Array.from({ length: totalEpochs }, (_, i) => {
    const epoch = i + 1;
    const base = 2.2 - 1.1 * (1 - Math.exp(-epoch / 6.5));
    const value = clamp(base + (rng() - 0.5) * 0.06, 0.9, 2.3);
    return { epoch, value };
  });

  const embeddingPoints = Array.from({ length: 110 }, (_, i) => {
    const label = i % 4;
    const center = label === 0 ? [-0.7, 0.2] : label === 1 ? [0.5, 0.7] : label === 2 ? [0.7, -0.4] : [-0.2, -0.7];
    const angle = rng() * Math.PI * 2;
    const radius = 0.3 + rng() * 0.45;
    const x = clamp(center[0] + Math.cos(angle) * radius, -1.2, 1.2);
    const y = clamp(center[1] + Math.sin(angle) * radius, -1.1, 1.1);
    return { id: i, x, y, label, weight: 0.6 + rng() * 1.0 };
  });

  const gradientPath = Array.from({ length: 28 }, (_, i) => {
    const t = i / 27;
    const x = clamp(0.45 - t * 0.45 + (rng() - 0.5) * 0.02, -1.1, 1.1);
    const y = clamp(-0.35 + t * 0.35 + (rng() - 0.5) * 0.02, -1.1, 1.1);
    return { x, y };
  });

  const residuals = Array.from({ length: 90 }, () => {
    const pred = clamp(0.2 + rng() * 0.8, 0, 1);
    const actual = clamp(pred + (rng() - 0.5) * 0.18, 0, 1);
    const residual = actual - pred;
    return { pred, true: actual, residual };
  });

  const leaderboard = [
    { rank: 1, model: 'RidgeRegression', metricName: 'rmse', metricValue: 0.82, params: { alpha: 0.7 } },
    { rank: 2, model: 'RandomForestRegressor', metricName: 'rmse', metricValue: 0.94, params: { n_estimators: 140 } },
    { rank: 3, model: 'SVR', metricName: 'rmse', metricValue: 1.05, params: { C: 3.2 } },
  ];

  const pipelineGraph = {
    nodes: [
      { id: 'ingest', label: 'Ingest' },
      { id: 'scale', label: 'Scale' },
      { id: 'train', label: 'Train' },
      { id: 'eval', label: 'Evaluate' },
    ],
    edges: [
      { from: 'ingest', to: 'scale' },
      { from: 'scale', to: 'train' },
      { from: 'train', to: 'eval' },
    ],
  };

  return {
    id: 'C',
    name: 'Regression (Ridge)',
    model: 'RidgeRegression',
    task: 'regression',
    totalEpochs,
    lossCurve,
    rmseCurve,
    embeddingPoints,
    gradientPath,
    residuals,
    leaderboard,
    pipelineGraph,
  };
};
