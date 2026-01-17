import { clamp, createRng, type ScenarioBuilder, type ScenarioData } from './scenarioTypes';

export const scenarioB: ScenarioBuilder = (seed = 2024): ScenarioData => {
  const rng = createRng(seed + 22);
  const totalEpochs = 26;

  const lossCurve = Array.from({ length: totalEpochs }, (_, i) => {
    const epoch = i + 1;
    const base = 1.65 * Math.exp(-epoch / 9.5) + 0.18;
    const noise = (rng() - 0.5) * 0.12;
    const train_loss = clamp(base + noise, 0.12, 2.0);
    const val_loss = clamp(base * 1.2 + noise * 0.9 + 0.15, 0.2, 2.2);
    return { epoch, train_loss, val_loss };
  });

  const accCurve = Array.from({ length: totalEpochs }, (_, i) => {
    const epoch = i + 1;
    const base = 0.38 + 0.38 * (1 - Math.exp(-epoch / 14));
    const value = clamp(base + (rng() - 0.5) * 0.04, 0.34, 0.82);
    return { epoch, value };
  });

  const confusion = [
    [280, 18, 12],
    [22, 260, 20],
    [15, 28, 250],
  ];

  const embeddingPoints = Array.from({ length: 150 }, (_, i) => {
    const label = i % 3;
    const center = label === 0 ? [-0.7, -0.1] : label === 1 ? [0.0, 0.7] : [0.75, -0.3];
    const angle = rng() * Math.PI * 2;
    const radius = 0.28 + rng() * 0.5;
    const x = clamp(center[0] + Math.cos(angle) * radius, -1.2, 1.2);
    const y = clamp(center[1] + Math.sin(angle) * radius, -1.1, 1.1);
    return { id: i, x, y, label, weight: 0.6 + rng() * 1.3 };
  });

  const gradientPath = Array.from({ length: 52 }, (_, i) => {
    const t = i / 51;
    const zig = Math.sin(t * Math.PI * 6) * 0.12;
    const jitter = (rng() - 0.5) * 0.14;
    const x = clamp(-0.9 + t * 1.3 + zig + jitter, -1.1, 1.1);
    const y = clamp(0.95 - t * 1.4 - zig * 0.8 + jitter * 0.6, -1.1, 1.1);
    return { x, y };
  });

  const leaderboard = [
    { rank: 1, model: 'RandomForest', metricName: 'macro_f1', metricValue: 0.84, params: { n_estimators: 220 } },
    { rank: 2, model: 'GradientBoosting', metricName: 'macro_f1', metricValue: 0.81, params: { learning_rate: 0.08 } },
    { rank: 3, model: 'LinearSVC', metricName: 'macro_f1', metricValue: 0.77, params: { C: 1.1 } },
  ];

  const pipelineGraph = {
    nodes: [
      { id: 'ingest', label: 'Ingest' },
      { id: 'profile', label: 'Profile' },
      { id: 'encode', label: 'Encode' },
      { id: 'train', label: 'Train' },
      { id: 'calibrate', label: 'Calibrate' },
      { id: 'eval', label: 'Evaluate' },
    ],
    edges: [
      { from: 'ingest', to: 'profile' },
      { from: 'profile', to: 'encode' },
      { from: 'encode', to: 'train' },
      { from: 'train', to: 'calibrate' },
      { from: 'calibrate', to: 'eval' },
    ],
  };

  return {
    id: 'B',
    name: 'Multiclass Classification (RandomForest)',
    model: 'RandomForest',
    task: 'multiclass',
    totalEpochs,
    lossCurve,
    accCurve,
    confusion,
    embeddingPoints,
    gradientPath,
    leaderboard,
    pipelineGraph,
  };
};
