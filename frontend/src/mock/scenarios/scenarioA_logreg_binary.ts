import { clamp, createRng, type ScenarioBuilder, type ScenarioData } from './scenarioTypes';

export const scenarioA: ScenarioBuilder = (seed = 1337): ScenarioData => {
  const rng = createRng(seed + 11);
  const totalEpochs = 24;

  const lossCurve = Array.from({ length: totalEpochs }, (_, i) => {
    const epoch = i + 1;
    const base = 1.05 * Math.exp(-epoch / 11) + 0.08;
    const noise = (rng() - 0.5) * 0.04;
    const train_loss = clamp(base + noise, 0.04, 1.2);
    const val_loss = clamp(base * 1.06 + noise * 0.7 + 0.03, 0.06, 1.3);
    return { epoch, train_loss, val_loss };
  });

  const accCurve = Array.from({ length: totalEpochs }, (_, i) => {
    const epoch = i + 1;
    const base = 0.62 + 0.28 * (1 - Math.exp(-epoch / 9));
    const value = clamp(base + (rng() - 0.5) * 0.015, 0.58, 0.93);
    return { epoch, value };
  });

  const confusion = [
    [620, 58],
    [72, 450],
  ];

  const embeddingPoints = Array.from({ length: 120 }, (_, i) => {
    const label = i % 2;
    const angle = rng() * Math.PI * 2;
    const radius = 0.35 + rng() * 0.45;
    const centerX = label === 0 ? -0.6 : 0.6;
    const centerY = label === 0 ? -0.2 : 0.3;
    const x = clamp(centerX + Math.cos(angle) * radius, -1.2, 1.2);
    const y = clamp(centerY + Math.sin(angle) * radius, -1.1, 1.1);
    return { id: i, x, y, label, weight: 0.6 + rng() * 1.1 };
  });

  const gradientPath = Array.from({ length: 48 }, (_, i) => {
    const t = i / 47;
    const curve = Math.sin(t * Math.PI) * 0.25;
    const x = clamp(0.95 - t * 1.35 + curve * 0.4, -1.1, 1.1);
    const y = clamp(0.85 - t * 1.05 - curve * 0.3, -1.1, 1.1);
    return { x, y };
  });

  const leaderboard = [
    { rank: 1, model: 'LogisticRegression', metricName: 'accuracy', metricValue: 0.89, params: { C: 1.2 } },
    { rank: 2, model: 'SVM', metricName: 'accuracy', metricValue: 0.86, params: { C: 0.9 } },
    { rank: 3, model: 'NaiveBayes', metricName: 'accuracy', metricValue: 0.81, params: { smoothing: 1.0 } },
  ];

  const pipelineGraph = {
    nodes: [
      { id: 'ingest', label: 'Ingest' },
      { id: 'clean', label: 'Clean' },
      { id: 'encode', label: 'Encode' },
      { id: 'train', label: 'Train' },
      { id: 'eval', label: 'Evaluate' },
    ],
    edges: [
      { from: 'ingest', to: 'clean' },
      { from: 'clean', to: 'encode' },
      { from: 'encode', to: 'train' },
      { from: 'train', to: 'eval' },
    ],
  };

  return {
    id: 'A',
    name: 'Binary Classification (LogReg)',
    model: 'LogisticRegression',
    task: 'binary',
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
