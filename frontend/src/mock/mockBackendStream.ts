import type { BackendEvent, LossSurfaceSpec, StepId } from './backendEventTypes';
import { MOCK_SCENARIOS, type ScenarioId, type ScenarioData } from './scenarios';

export interface MockStreamOptions {
  scenarioId?: ScenarioId;
  seed?: number;
  speed?: number; // >1 faster, <1 slower
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function now() {
  return Date.now();
}

function createRng(seed = 1337) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function buildSurfaceSpec(scenarioId: ScenarioId, seed?: number): LossSurfaceSpec {
  const rng = createRng((seed ?? 1337) + scenarioId.charCodeAt(0) * 97);
  if (scenarioId === 'A') {
    return {
      kind: 'bowl',
      domainHalf: 6,
      zScale: 0.3,
      params: { a: 0.18 + rng() * 0.01, b: 0.17 + rng() * 0.01, tiltX: 0.02, tiltY: -0.01, offset: -3 },
    };
  }
  if (scenarioId === 'B') {
    return {
      kind: 'multi_hill',
      domainHalf: 6,
      zScale: 0.3,
      params: {
        bowlStrength: 0.07,
        offset: -3,
        hills: [
          { x: -2.2, y: 1.5, amp: 6.5, sigma: 2.2 },
          { x: 1.6, y: -1.8, amp: 5.0, sigma: 1.7 },
          { x: 0.3, y: 2.5, amp: 3.8, sigma: 1.2 },
        ],
      },
    };
  }
  return {
    kind: 'ripples',
    domainHalf: 6,
    zScale: 0.3,
    params: { amp: 2.7, freq: 2.2, decay: 0.15, bowlStrength: 0.06, offset: -3 },
  };
}

function toMissingness(columns: string[]) {
  return columns.map((column, i) => ({
    column,
    missingPct: Math.round(((i + 1) * 7) % 30) / 100,
  }));
}

function buildArtifacts(scenario: ScenarioData) {
  const task = {
    task: scenario.task,
    model: scenario.model,
    target: scenario.task === 'regression' ? 'target' : 'churn',
  };

  const schema = {
    rows: 70430,
    columns: ['tenure', 'monthly_charges', 'contract_type', 'payment_method', 'churn'],
  };

  const features = {
    total: 38,
    top: ['monthly_charges', 'tenure', 'contract_type'],
  };

  const modelPlan = {
    selected: scenario.model,
    candidates: scenario.leaderboard.map((e) => ({ model: e.model, params: e.params })),
  };

  const metrics = {
    loss_curve: scenario.lossCurve,
    accuracy_curve: scenario.accCurve,
    rmse_curve: scenario.rmseCurve,
    confusion: scenario.confusion,
  };

  const report = {
    summary: `${scenario.model} report summary`,
    highlights: ['Stable convergence', 'No leakage detected'],
  };

  const modelCard = `# Model Card\n\nModel: ${scenario.model}\nTask: ${scenario.task}\n`;

  return {
    task,
    schema,
    features,
    modelPlan,
    metrics,
    report,
    modelCard,
  };
}

export async function* createMockAutoMLStream(options: MockStreamOptions = {}): AsyncGenerator<BackendEvent> {
  const scenarioId = options.scenarioId ?? 'A';
  // CHANGE SCENARIO HERE: scenarioId controls which mock ML run is streamed.
  const scenario = MOCK_SCENARIOS[scenarioId].build(options.seed ?? 1337);
  const surfaceSpec = buildSurfaceSpec(scenarioId, options.seed);
  const speed = options.speed ?? 1;
  const delay = (ms: number) => sleep(ms / speed);

  const stepEvent = (step: StepId, status: 'waiting' | 'running' | 'complete', progress?: number, message?: string): BackendEvent => ({
    type: 'STEP_STATUS',
    step,
    status,
    progress,
    message,
    ts: now(),
  });

  const artifacts = buildArtifacts(scenario);

  yield { type: 'LOSS_SURFACE_SPEC', spec: surfaceSpec, ts: now() };
  await delay(140);
  yield { type: 'GD_PATH', points: scenario.gradientPath, ts: now() };
  await delay(160);
  yield {
    type: 'ARTIFACT_WRITTEN',
    path: 'artifacts/gradient_path.json',
    content: JSON.stringify({ path: scenario.gradientPath }, null, 2),
    ts: now(),
  };

  // S0 Task Spec
  yield stepEvent('S0', 'running', 0.1, 'Interpreting task');
  yield {
    type: 'PLAN_PROPOSED',
    step: 'S0',
    planId: 'task_plan',
    variants: [
      {
        id: 'task_v1',
        title: 'Task Understanding v1',
        description: `AutoML ${scenario.task} plan with ${scenario.model}`,
        models: [scenario.model],
        expectedMinutes: 5,
      },
    ],
    ts: now(),
  };
  yield stepEvent('S0', 'complete', 1);
  yield { type: 'ARTIFACT_WRITTEN', path: 'config/task.json', content: JSON.stringify(artifacts.task, null, 2), ts: now() };
  await delay(300);

  // S1 Data Source
  yield stepEvent('S1', 'running', 0.1, 'Searching datasets');
  yield {
    type: 'DATASET_SEARCH_RESULTS',
    query: 'customer churn',
    results: [
      { id: 'ds_a', name: 'Telecom Churn', license: 'CC BY 4.0', sizeMB: 68, columns: 24 },
      { id: 'ds_b', name: 'SaaS Churn', license: 'MIT', sizeMB: 28, columns: 18 },
      { id: 'ds_c', name: 'Bank Churn', license: 'ODC-By', sizeMB: 42, columns: 21 },
    ],
    ts: now(),
  };
  yield { type: 'DATASET_INGESTED', datasetId: 'ds_a', rows: 70430, columns: 24, ts: now() };
  yield stepEvent('S1', 'complete', 1);
  await delay(300);

  // S2 Profiling + Schema
  yield stepEvent('S2', 'running', 0.2, 'Profiling schema');
  yield { type: 'PROFILE_PROGRESS', stage: 'Scanning columns', progress: 0.4, ts: now() };
  yield { type: 'PROFILE_PROGRESS', stage: 'Computing missingness', progress: 0.8, ts: now() };
  const missingness = toMissingness(['tenure', 'monthly_charges', 'contract_type', 'payment_method', 'churn']);
  yield { type: 'PROFILE_SUMMARY', rows: 70430, columns: 24, missingness, ts: now() };
  yield {
    type: 'METRIC_TABLE',
    table: 'missingness',
    rows: missingness.map((m) => m.column),
    cols: ['missingPct'],
    data: missingness.map((m) => [m.missingPct]),
    ts: now(),
  };
  yield { type: 'ARTIFACT_WRITTEN', path: 'config/schema.json', content: JSON.stringify(artifacts.schema, null, 2), ts: now() };
  yield stepEvent('S2', 'complete', 1);
  await delay(350);

  // S3 Feature Plan
  yield stepEvent('S3', 'running', 0.2, 'Designing features');
  yield {
    type: 'PLAN_PROPOSED',
    step: 'S3',
    planId: 'feature_plan',
    variants: [
      { id: 'feat_a', title: 'Balanced', description: 'Standard encoders + scaling', models: [scenario.model], expectedMinutes: 12 },
      { id: 'feat_b', title: 'High Recall', description: 'Feature expansion + interactions', models: [scenario.model], expectedMinutes: 18 },
    ],
    ts: now(),
  };
  yield { type: 'PLAN_SELECTED', step: 'S3', planId: 'feature_plan', variantId: 'feat_a', ts: now() };
  yield { type: 'PIPELINE_GRAPH', nodes: scenario.pipelineGraph.nodes, edges: scenario.pipelineGraph.edges, ts: now() };
  yield { type: 'FEATURE_SUMMARY', totalFeatures: 38, topFeatures: [{ name: 'tenure', importance: 0.22 }], ts: now() };
  yield { type: 'ARTIFACT_WRITTEN', path: 'config/features.json', content: JSON.stringify(artifacts.features, null, 2), ts: now() };
  yield stepEvent('S3', 'complete', 1);
  await delay(350);

  // S4 Model Plan
  yield stepEvent('S4', 'running', 0.2, 'Selecting model');
  yield {
    type: 'PLAN_PROPOSED',
    step: 'S4',
    planId: 'model_plan',
    variants: scenario.leaderboard.map((entry, idx) => ({
      id: `model_${idx + 1}`,
      title: entry.model,
      description: `Expected ${entry.metricName}: ${entry.metricValue.toFixed(2)}`,
      models: [entry.model],
      expectedMinutes: 10 + idx * 6,
    })),
    ts: now(),
  };
  yield { type: 'PLAN_SELECTED', step: 'S4', planId: 'model_plan', variantId: 'model_1', ts: now() };
  yield { type: 'ARTIFACT_WRITTEN', path: 'config/model_plan.json', content: JSON.stringify(artifacts.modelPlan, null, 2), ts: now() };
  yield { type: 'LEADERBOARD_UPDATED', entries: scenario.leaderboard, ts: now() };
  yield stepEvent('S4', 'complete', 1);
  await delay(350);

  // S5 Train/HPO
  yield stepEvent('S5', 'running', 0.1, 'Training');
  for (const point of scenario.lossCurve) {
    yield { type: 'TRAIN_PROGRESS', epoch: point.epoch, totalEpochs: scenario.totalEpochs, ts: now() };
    yield { type: 'METRIC_SCALAR', metric: 'train_loss', epoch: point.epoch, value: point.train_loss, split: 'train', ts: now() };
    yield { type: 'METRIC_SCALAR', metric: 'val_loss', epoch: point.epoch, value: point.val_loss, split: 'val', ts: now() };

    const metricSeries = scenario.task === 'regression' ? scenario.rmseCurve : scenario.accCurve;
    const metric = metricSeries?.find((m) => m.epoch === point.epoch);
    if (metric) {
      yield {
        type: 'METRIC_SCALAR',
        metric: scenario.task === 'regression' ? 'rmse' : 'accuracy',
        epoch: metric.epoch,
        value: metric.value,
        ts: now(),
      };
      if (scenario.task !== 'regression') {
        yield { type: 'METRIC_SCALAR', metric: 'f1', epoch: metric.epoch, value: Math.max(0.6, metric.value - 0.05), ts: now() };
      }
    }
    yield { type: 'LOG_LINE', level: 'INFO', message: `Epoch ${point.epoch}/${scenario.totalEpochs} complete`, ts: now() };
    await delay(220);
  }

  if (scenario.confusion) {
    const labels = scenario.confusion.map((_, idx) => `Class ${idx + 1}`);
    yield { type: 'METRIC_TABLE', table: 'confusion', rows: labels, cols: labels, data: scenario.confusion, ts: now() };
  }
  yield { type: 'LEADERBOARD_UPDATED', entries: scenario.leaderboard, ts: now() };
  yield { type: 'ARTIFACT_WRITTEN', path: 'artifacts/metrics.json', content: JSON.stringify(artifacts.metrics, null, 2), ts: now() };
  yield { type: 'ARTIFACT_WRITTEN', path: 'artifacts/model.joblib', content: 'BASE64:mock-model', ts: now() };
  yield { type: 'ARTIFACT_WRITTEN', path: 'artifacts/embedding_points.json', content: JSON.stringify(scenario.embeddingPoints, null, 2), ts: now() };
  if (scenario.residuals && scenario.residuals.length > 0) {
    yield {
      type: 'ARTIFACT_WRITTEN',
      path: 'artifacts/residuals.json',
      content: JSON.stringify({ points: scenario.residuals }, null, 2),
      ts: now(),
    };
  }
  yield stepEvent('S5', 'complete', 1);
  await delay(350);

  // S6 Report
  yield stepEvent('S6', 'running', 0.4, 'Writing report');
  yield { type: 'REPORT_READY', summary: `Report ready for ${scenario.model}`, ts: now() };
  yield { type: 'ARTIFACT_WRITTEN', path: 'artifacts/report.json', content: JSON.stringify(artifacts.report, null, 2), ts: now() };
  yield { type: 'ARTIFACT_WRITTEN', path: 'artifacts/model_card.md', content: artifacts.modelCard, ts: now() };
  yield stepEvent('S6', 'complete', 1);
  await delay(300);

  // S7 Export
  yield stepEvent('S7', 'running', 0.7, 'Exporting');
  yield { type: 'EXPORT_READY', files: ['artifacts/model.joblib', 'artifacts/model_card.md'], ts: now() };
  yield stepEvent('S7', 'complete', 1);
}

/*
import type {
  BackendEvent,
  DatasetResult,
  FeatureSummary,
  LeaderboardEntry,
  PipelineGraph,
  PlanVariant,
  StepId,
  StepStatus,
} from '../types/backendEvents';

export interface MockStreamOptions {
  seed?: number;
  speed?: number; // >1 faster, <1 slower
  autoStart?: boolean;
}

export interface MockBackendApi {
  confirmStep: (stepId: StepId) => void;
  selectPlan: (stepId: StepId, planId: string) => void;
  pause: () => void;
  resume: () => void;
  stop: () => void;
}

export interface MockStreamResult {
  stream: AsyncIterable<BackendEvent>;
  api: MockBackendApi;
}

class AsyncEventQueue<T> implements AsyncIterable<T> {
  private queue: T[] = [];
  private resolvers: Array<(value: IteratorResult<T>) => void> = [];
  private closed = false;

  push(value: T) {
    if (this.closed) return;
    const resolver = this.resolvers.shift();
    if (resolver) {
      resolver({ value, done: false });
    } else {
      this.queue.push(value);
    }
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    while (this.resolvers.length) {
      const resolver = this.resolvers.shift();
      if (resolver) resolver({ value: undefined as T, done: true });
    }
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: () => {
        if (this.queue.length) {
          const value = this.queue.shift() as T;
          return Promise.resolve({ value, done: false });
        }
        if (this.closed) return Promise.resolve({ value: undefined as T, done: true });
        return new Promise((resolve) => this.resolvers.push(resolve));
      },
    };
  }
}

const defaultSpeed = 1;

function createRng(seed?: number) {
  if (seed == null) return () => Math.random();
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(num: number, min: number, max: number) {
  return Math.max(min, Math.min(max, num));
}

export function createMockAutoMLStream(options: MockStreamOptions = {}): MockStreamResult {
  const queue = new AsyncEventQueue<BackendEvent>();
  const rng = createRng(options.seed);
  const speed = options.speed ?? defaultSpeed;

  let paused = false;
  let stopped = false;
  const resumeWaiters: Array<() => void> = [];

  const confirmWaiters = new Map<StepId, () => void>();
  const planWaiters = new Map<StepId, (planId: string) => void>();
  const selectedPlans = new Map<StepId, string>();

  const api: MockBackendApi = {
    confirmStep(stepId) {
      const resolve = confirmWaiters.get(stepId);
      if (resolve) {
        confirmWaiters.delete(stepId);
        resolve();
      }
    },
    selectPlan(stepId, planId) {
      selectedPlans.set(stepId, planId);
      const resolve = planWaiters.get(stepId);
      if (resolve) {
        planWaiters.delete(stepId);
        resolve(planId);
      }
    },
    pause() {
      paused = true;
    },
    resume() {
      paused = false;
      while (resumeWaiters.length) {
        const resolve = resumeWaiters.shift();
        if (resolve) resolve();
      }
    },
    stop() {
      stopped = true;
      queue.close();
    },
  };

  const waitWhilePaused = () =>
    new Promise<void>((resolve) => {
      resumeWaiters.push(resolve);
    });

  const wait = async (ms: number) => {
    let remaining = Math.max(0, ms / speed);
    while (remaining > 0) {
      if (stopped) throw new Error('stopped');
      if (paused) {
        await waitWhilePaused();
        continue;
      }
      const slice = Math.min(200, remaining);
      await new Promise((resolve) => setTimeout(resolve, slice));
      remaining -= slice;
    }
  };

  const waitForConfirm = (step: StepId) =>
    new Promise<void>((resolve) => {
      confirmWaiters.set(step, resolve);
    });

  const waitForPlan = (step: StepId) =>
    new Promise<string>((resolve) => {
      const existing = selectedPlans.get(step);
      if (existing) {
        resolve(existing);
        return;
      }
      planWaiters.set(step, resolve);
    });

  const emit = (event: BackendEvent) => {
    if (stopped) return;
    queue.push(event);
  };

  const emitStepStatus = (step: StepId, status: StepStatus, progress?: number, message?: string) => {
    emit({
      type: 'STEP_STATUS',
      ts: Date.now(),
      step,
      payload: { status, progress, message },
    });
  };

  const pick = <T,>(arr: T[]) => arr[Math.floor(rng() * arr.length)];

  const buildMissingness = (columns: string[]) =>
    columns.map((column) => ({
      column,
      missingPct: Math.round(clamp(rng() * 0.3, 0, 0.3) * 1000) / 1000,
    }));

  const buildConfusionMatrix = (n: number, accuracy: number) => {
    const correct = Math.round(n * accuracy);
    const incorrect = n - correct;
    const tp = Math.round(correct * 0.55);
    const tn = correct - tp;
    const fp = Math.round(incorrect * 0.6);
    const fn = incorrect - fp;
    return {
      columns: ['Pred 0', 'Pred 1'],
      rows: [
        { 'Actual 0': tn, 'Pred 1': fp },
        { 'Actual 1': fn, 'Pred 1': tp },
      ],
      matrix: [
        [tn, fp],
        [fn, tp],
      ],
      n,
    };
  };

  const buildLeaderboard = (entries: Array<Omit<LeaderboardEntry, 'rank'>>) => {
    const sorted = [...entries].sort((a, b) => b.metric - a.metric);
    return sorted.map((entry, idx) => ({ ...entry, rank: idx + 1 }));
  };

  const runTimeline = async () => {
    try {
      // S0 Task Spec
      emitStepStatus('S0', 'running', 0.05, 'Analyzing task requirements');
      const taskPlan: PlanVariant = {
        id: 'task_plan_v1',
        title: 'Task Understanding v1',
        summary: 'Binary classification on customer churn with 24 features.',
        details: { target: 'churn', inputType: 'tabular' },
      };
      emit({
        type: 'PLAN_PROPOSED',
        ts: Date.now(),
        step: 'S0',
        payload: { plans: [taskPlan], recommendedPlanId: taskPlan.id },
      });
      emitStepStatus('S0', 'waiting_confirmation', 0.1, 'Awaiting task spec confirmation');
      await waitForConfirm('S0');
      emit({
        type: 'ARTIFACT_WRITTEN',
        ts: Date.now(),
        step: 'S0',
        payload: {
          path: 'config/task.json',
          contentType: 'json',
          content: JSON.stringify(
            {
              task_type: 'classification',
              target: 'churn',
              objective: 'maximize_f1',
              constraints: { max_runtime_min: 45 },
            },
            null,
            2
          ),
        },
      });
      emitStepStatus('S0', 'completed', 1);

      await wait(400);

      // S1 Data Source Selection
      emitStepStatus('S1', 'running', 0.1, 'Searching datasets');
      const datasetResults: DatasetResult[] = [
        {
          id: 'ds_telecom_churn_v3',
          name: 'Telecom Customer Churn 2024',
          license: 'CC BY 4.0',
          sizeMB: 68,
          rows: 70430,
          columns: [
            { name: 'tenure', type: 'number', missingPct: 0.02 },
            { name: 'monthly_charges', type: 'number', missingPct: 0.01 },
            { name: 'contract_type', type: 'category', missingPct: 0.0 },
          ],
          description: 'Telco churn dataset with contract and usage signals.',
        },
        {
          id: 'ds_banking_churn',
          name: 'Retail Banking Churn Signals',
          license: 'ODC-By',
          sizeMB: 42,
          rows: 50213,
          columns: [
            { name: 'credit_score', type: 'number', missingPct: 0.03 },
            { name: 'products', type: 'number', missingPct: 0.0 },
            { name: 'has_cr_card', type: 'boolean', missingPct: 0.0 },
          ],
          description: 'Banking churn with demographic + product usage.',
        },
        {
          id: 'ds_saas_churn',
          name: 'SaaS Subscription Churn',
          license: 'MIT',
          sizeMB: 28,
          rows: 28990,
          columns: [
            { name: 'seats', type: 'number', missingPct: 0.08 },
            { name: 'mrr', type: 'number', missingPct: 0.01 },
            { name: 'plan_tier', type: 'category', missingPct: 0.0 },
          ],
          description: 'Subscription churn with revenue + engagement signals.',
        },
      ];
      emit({
        type: 'DATASET_SEARCH_RESULTS',
        ts: Date.now(),
        step: 'S1',
        payload: { query: 'customer churn classification', results: datasetResults },
      });

      const datasetPlans: PlanVariant[] = datasetResults.map((ds, idx) => ({
        id: `dataset_plan_${idx + 1}`,
        title: `Use ${ds.name}`,
        summary: ds.description,
        details: { datasetId: ds.id, rows: ds.rows, sizeMB: ds.sizeMB },
      }));

      emit({
        type: 'PLAN_PROPOSED',
        ts: Date.now(),
        step: 'S1',
        payload: { plans: datasetPlans, recommendedPlanId: datasetPlans[0].id },
      });
      emitStepStatus('S1', 'waiting_confirmation', 0.2, 'Select dataset and confirm');
      const selectedDatasetPlan = await waitForPlan('S1');
      emit({
        type: 'PLAN_SELECTED',
        ts: Date.now(),
        step: 'S1',
        payload: { planId: selectedDatasetPlan, reason: 'Best coverage for churn features' },
      });
      const selectedDataset = datasetPlans.find((p) => p.id === selectedDatasetPlan)?.details?.datasetId as string;
      emit({
        type: 'DATASET_INGESTED',
        ts: Date.now(),
        step: 'S1',
        payload: { datasetId: selectedDataset || datasetResults[0].id, rows: 70430, columns: 24 },
      });
      await waitForConfirm('S1');
      emitStepStatus('S1', 'completed', 1);

      await wait(500);

      // S2 Profiling + Schema
      emitStepStatus('S2', 'running', 0.05, 'Profiling dataset');
      const profileStages = ['Scanning columns', 'Detecting types', 'Computing missingness', 'Validating target'];
      for (let i = 0; i < profileStages.length; i += 1) {
        emit({
          type: 'PROFILE_PROGRESS',
          ts: Date.now(),
          step: 'S2',
          payload: { stage: profileStages[i], progress: (i + 1) / profileStages.length },
        });
        emit({
          type: 'RESOURCE_STATS',
          ts: Date.now(),
          step: 'S2',
          payload: {
            cpuPct: Math.round(20 + rng() * 15),
            ramMB: Math.round(1400 + rng() * 300),
            gpuPct: Math.round(5 + rng() * 10),
          },
        });
        await wait(420);
      }

      const schemaColumns = ['tenure', 'monthly_charges', 'contract_type', 'payment_method', 'churn'];
      const missingness = buildMissingness(schemaColumns);

      emit({
        type: 'PROFILE_SUMMARY',
        ts: Date.now(),
        step: 'S2',
        payload: { rows: 70430, columns: 24, missingness, target: 'churn' },
      });

      emit({
        type: 'METRIC_TABLE',
        ts: Date.now(),
        step: 'S2',
        payload: {
          name: 'missingness',
          columns: ['column', 'missingPct'],
          rows: missingness.map((m) => ({ column: m.column, missingPct: m.missingPct })),
        },
      });

      emit({
        type: 'ARTIFACT_WRITTEN',
        ts: Date.now(),
        step: 'S2',
        payload: {
          path: 'config/schema.json',
          contentType: 'json',
          content: JSON.stringify(
            {
              rows: 70430,
              columns: schemaColumns.map((name) => ({
                name,
                dtype: name === 'churn' ? 'boolean' : 'string',
                missing: missingness.find((m) => m.column === name)?.missingPct ?? 0,
              })),
              target: 'churn',
            },
            null,
            2
          ),
        },
      });

      emit({
        type: 'PLAN_PROPOSED',
        ts: Date.now(),
        step: 'S2',
        payload: {
          plans: [
            {
              id: 'profile_plan_v1',
              title: 'Profile & schema v1',
              summary: 'Proceed with detected schema + missingness remediation',
            },
          ],
          recommendedPlanId: 'profile_plan_v1',
        },
      });
      emitStepStatus('S2', 'waiting_confirmation', 0.9, 'Confirm profiling summary');
      await waitForConfirm('S2');
      emitStepStatus('S2', 'completed', 1);

      await wait(600);

      // S3 Feature Plan
      emitStepStatus('S3', 'running', 0.1, 'Designing feature pipeline');
      const featurePlans: PlanVariant[] = [
        {
          id: 'feat_plan_a',
          title: 'Balanced feature plan',
          summary: 'Standard encoders + numeric scaling + target mean encoding.',
        },
        {
          id: 'feat_plan_b',
          title: 'High-recall plan',
          summary: 'Aggressive categorical expansion + interaction terms.',
        },
        {
          id: 'feat_plan_c',
          title: 'Fast baseline plan',
          summary: 'Minimal preprocessing for rapid iteration.',
        },
      ];
      emit({
        type: 'PLAN_PROPOSED',
        ts: Date.now(),
        step: 'S3',
        payload: { plans: featurePlans, recommendedPlanId: 'feat_plan_a' },
      });
      emitStepStatus('S3', 'waiting_confirmation', 0.2, 'Select feature plan');
      const selectedFeaturePlan = await waitForPlan('S3');
      emit({
        type: 'PLAN_SELECTED',
        ts: Date.now(),
        step: 'S3',
        payload: { planId: selectedFeaturePlan, reason: 'Balanced performance and explainability' },
      });

      const graph: PipelineGraph = {
        nodes: [
          { id: 'ingest', label: 'Ingest', kind: 'data' },
          { id: 'clean', label: 'Clean', kind: 'transform' },
          { id: 'encode', label: 'Encode', kind: 'transform' },
          { id: 'model', label: 'Model', kind: 'train' },
        ],
        edges: [
          { from: 'ingest', to: 'clean' },
          { from: 'clean', to: 'encode' },
          { from: 'encode', to: 'model' },
        ],
      };

      const featureSummary: FeatureSummary = {
        totalFeatures: 38,
        topFeatures: [
          { name: 'monthly_charges', importance: 0.28 },
          { name: 'tenure', importance: 0.22 },
          { name: 'contract_type', importance: 0.14 },
        ],
        encoding: ['one_hot', 'target_mean'],
      };

      emit({
        type: 'PIPELINE_GRAPH',
        ts: Date.now(),
        step: 'S3',
        payload: { graph },
      });
      emit({
        type: 'FEATURE_SUMMARY',
        ts: Date.now(),
        step: 'S3',
        payload: { summary: featureSummary },
      });
      emit({
        type: 'FILE_UPDATED',
        ts: Date.now(),
        step: 'S3',
        payload: { path: 'src/features.py', message: 'Generated feature pipeline' },
      });
      emit({
        type: 'ARTIFACT_WRITTEN',
        ts: Date.now(),
        step: 'S3',
        payload: {
          path: 'config/features.json',
          contentType: 'json',
          content: JSON.stringify(featureSummary, null, 2),
        },
      });
      emitStepStatus('S3', 'waiting_confirmation', 0.8, 'Confirm feature plan');
      await waitForConfirm('S3');
      emitStepStatus('S3', 'completed', 1);

      await wait(600);

      // S4 Model Plan
      emitStepStatus('S4', 'running', 0.1, 'Evaluating model candidates');
      const modelPlans: PlanVariant[] = [
        {
          id: 'model_plan_a',
          title: 'Logistic Regression + Calibration',
          summary: 'Fast baseline with interpretable coefficients',
          expectedRuntimeMin: 6,
          interpretability: 'High',
        },
        {
          id: 'model_plan_b',
          title: 'Random Forest',
          summary: 'Balanced performance with feature importances',
          expectedRuntimeMin: 18,
          interpretability: 'Medium',
        },
        {
          id: 'model_plan_c',
          title: 'XGBoost',
          summary: 'Best accuracy with moderate interpretability',
          expectedRuntimeMin: 28,
          interpretability: 'Medium',
        },
      ];
      emit({
        type: 'PLAN_PROPOSED',
        ts: Date.now(),
        step: 'S4',
        payload: { plans: modelPlans, recommendedPlanId: 'model_plan_c' },
      });
      emitStepStatus('S4', 'waiting_confirmation', 0.2, 'Select model plan');
      const selectedModelPlan = await waitForPlan('S4');
      emit({
        type: 'PLAN_SELECTED',
        ts: Date.now(),
        step: 'S4',
        payload: { planId: selectedModelPlan, reason: 'Projected best accuracy' },
      });
      emit({
        type: 'ARTIFACT_WRITTEN',
        ts: Date.now(),
        step: 'S4',
        payload: {
          path: 'config/model_plan.json',
          contentType: 'json',
          content: JSON.stringify(
            {
              selectedPlan: selectedModelPlan,
              candidates: modelPlans,
            },
            null,
            2
          ),
        },
      });
      emit({
        type: 'LEADERBOARD_UPDATED',
        ts: Date.now(),
        step: 'S4',
        payload: { metric: 'f1', entries: [] },
      });
      await waitForConfirm('S4');
      emitStepStatus('S4', 'completed', 1);

      await wait(600);

      // S5 Train/HPO
      emitStepStatus('S5', 'running', 0.1, 'Training models');
      const runs = [
        { id: 'run_lr', name: 'LogReg', base: 0.78 },
        { id: 'run_rf', name: 'RandomForest', base: 0.84 },
        { id: 'run_xgb', name: 'XGBoost', base: 0.88 },
      ];
      const leaderboard: Array<Omit<LeaderboardEntry, 'rank'>> = [];

      for (const run of runs) {
        for (let p = 1; p <= 5; p += 1) {
          const progress = p / 5;
          const accuracy = clamp(run.base + rng() * 0.02 + progress * 0.015, 0, 0.95);
          const f1 = clamp(accuracy - 0.03 + rng() * 0.015, 0, 0.95);

          emit({
            type: 'TRAIN_PROGRESS',
            ts: Date.now(),
            step: 'S5',
            payload: { runId: run.id, modelName: run.name, progress, etaSec: Math.round((1 - progress) * 35) },
          });
          emit({
            type: 'METRIC_SCALAR',
            ts: Date.now(),
            step: 'S5',
            payload: { name: 'accuracy', value: accuracy, runId: run.id, stepIndex: p },
          });
          emit({
            type: 'METRIC_SCALAR',
            ts: Date.now(),
            step: 'S5',
            payload: { name: 'f1', value: f1, runId: run.id, stepIndex: p },
          });
          emit({
            type: 'RESOURCE_STATS',
            ts: Date.now(),
            step: 'S5',
            payload: {
              cpuPct: Math.round(45 + rng() * 20),
              ramMB: Math.round(2400 + rng() * 500),
              gpuPct: Math.round(15 + rng() * 25),
            },
          });
          emit({
            type: 'LOG_LINE',
            ts: Date.now(),
            step: 'S5',
            payload: { level: 'info', message: `${run.name} epoch ${p}/5 complete` },
          });
          await wait(420);
        }
        const finalMetric = clamp(run.base + 0.04 + rng() * 0.02, 0, 0.96);
        leaderboard.push({
          modelId: run.id,
          modelName: run.name,
          metric: finalMetric,
          params: { depth: pick([4, 6, 8]), lr: Math.round((0.05 + rng() * 0.1) * 100) / 100 },
        });
      }

      const sortedLeaderboard = buildLeaderboard(leaderboard);
      emit({
        type: 'LEADERBOARD_UPDATED',
        ts: Date.now(),
        step: 'S5',
        payload: { metric: 'accuracy', entries: sortedLeaderboard },
      });

      const best = sortedLeaderboard[0];
      const confusion = buildConfusionMatrix(1200, best?.metric ?? 0.9);
      emit({
        type: 'METRIC_TABLE',
        ts: Date.now(),
        step: 'S5',
        payload: {
          name: 'confusion_matrix',
          columns: ['Pred 0', 'Pred 1'],
          rows: [
            { label: 'Actual 0', 'Pred 0': confusion.matrix[0][0], 'Pred 1': confusion.matrix[0][1] },
            { label: 'Actual 1', 'Pred 0': confusion.matrix[1][0], 'Pred 1': confusion.matrix[1][1] },
          ],
        },
      });

      emit({
        type: 'ARTIFACT_WRITTEN',
        ts: Date.now(),
        step: 'S5',
        payload: {
          path: 'artifacts/metrics.json',
          contentType: 'json',
          content: JSON.stringify(
            {
              leaderboard: sortedLeaderboard,
              best_model: best?.modelName,
              accuracy: best?.metric,
              confusion_matrix: confusion.matrix,
            },
            null,
            2
          ),
        },
      });
      emit({
        type: 'ARTIFACT_WRITTEN',
        ts: Date.now(),
        step: 'S5',
        payload: {
          path: 'artifacts/model.joblib',
          contentType: 'binary',
          content: 'BASE64:UEtUR...mock-model-binary',
        },
      });
      emitStepStatus('S5', 'waiting_confirmation', 0.95, 'Confirm training results');
      await waitForConfirm('S5');
      emitStepStatus('S5', 'completed', 1);

      await wait(600);

      // S6 Evaluate/Report
      emitStepStatus('S6', 'running', 0.1, 'Generating report');
      emit({
        type: 'REPORT_READY',
        ts: Date.now(),
        step: 'S6',
        payload: {
          path: 'artifacts/report.json',
          highlights: ['Best model: XGBoost', 'Accuracy 0.91', 'Top feature: monthly_charges'],
          verificationNotes: ['No leakage detected', 'Train/val split 70/15/15'],
        },
      });
      emit({
        type: 'ARTIFACT_WRITTEN',
        ts: Date.now(),
        step: 'S6',
        payload: {
          path: 'artifacts/report.json',
          contentType: 'json',
          content: JSON.stringify(
            {
              summary: 'Model performance report',
              recommendations: ['Monitor drift monthly', 'Re-train quarterly'],
              verification: ['No leakage detected'],
            },
            null,
            2
          ),
        },
      });
      emit({
        type: 'ARTIFACT_WRITTEN',
        ts: Date.now(),
        step: 'S6',
        payload: {
          path: 'artifacts/model_card.md',
          contentType: 'text',
          content: '# Model Card\n\n- Task: Churn classification\n- Best model: XGBoost\n- Intended use: churn risk alerts',
        },
      });
      emitStepStatus('S6', 'waiting_confirmation', 0.9, 'Confirm evaluation report');
      await waitForConfirm('S6');
      emitStepStatus('S6', 'completed', 1);

      await wait(600);

      // S7 Export/Deploy
      emitStepStatus('S7', 'running', 0.1, 'Preparing export');
      emit({
        type: 'EXPORT_READY',
        ts: Date.now(),
        step: 'S7',
        payload: {
          targets: ['REST', 'Batch', 'ONNX'],
          files: ['artifacts/model.joblib', 'artifacts/serving_spec.json', 'artifacts/model.onnx'],
          notes: 'ONNX export optional for edge deployment',
        },
      });
      emit({
        type: 'ARTIFACT_WRITTEN',
        ts: Date.now(),
        step: 'S7',
        payload: {
          path: 'artifacts/serving_spec.json',
          contentType: 'json',
          content: JSON.stringify(
            {
              endpoint: '/predict',
              method: 'POST',
              inputs: ['tenure', 'monthly_charges', 'contract_type'],
              outputs: ['churn_probability'],
            },
            null,
            2
          ),
        },
      });
      emitStepStatus('S7', 'waiting_confirmation', 0.95, 'Confirm export');
      await waitForConfirm('S7');
      emitStepStatus('S7', 'completed', 1);
    } catch (error) {
      // Stop gracefully
    } finally {
      queue.close();
    }
  };

  if (options.autoStart !== false) {
    void runTimeline();
  }

  return { stream: queue, api };
}

export function toSSE(stream: AsyncIterable<BackendEvent>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async start(controller) {
      for await (const event of stream) {
        const data = `event: ${event.type}\n` + `data: ${JSON.stringify(event)}\n\n`;
        controller.enqueue(encoder.encode(data));
      }
      controller.close();
    },
  });
}
*/
