import type { ScenarioBuilder, ScenarioId } from './scenarioTypes';
import { scenarioA } from './scenarioA_logreg_binary';
import { scenarioB } from './scenarioB_rf_multiclass';
import { scenarioC } from './scenarioC_ridge_regression';

export type { ScenarioId } from './scenarioTypes';
export type { ScenarioData, ScenarioBuilder } from './scenarioTypes';

export const MOCK_SCENARIOS: Record<ScenarioId, { name: string; model: string; build: ScenarioBuilder }> = {
  A: { name: 'Binary Classification (LogReg)', model: 'LogisticRegression', build: scenarioA },
  B: { name: 'Multiclass Classification (RandomForest)', model: 'RandomForest', build: scenarioB },
  C: { name: 'Regression (Ridge)', model: 'RidgeRegression', build: scenarioC },
};
