import React from 'react';
import type { VisualId, VisualProps } from '../types';
import NeuralNetViz from './NeuralNetViz';
import GradientDescentViz from './GradientDescentViz';
import EvaluationViz from './EvaluationViz';
import ConfusionMatrixViz from './ConfusionMatrixViz';
import EmbeddingScatterViz from './EmbeddingScatterViz';
import ResidualsViz from './ResidualsViz';

export const VISUAL_LABEL: Record<VisualId, string> = {
  neuralNetForward: 'Neural Net (Forward)',
  neuralNetBackprop: 'Neural Net (Backprop)',
  gradDescent: 'Gradient Descent',
  evaluation: 'Evaluation Suite',
  confusionMatrix: 'Confusion Matrix',
  embeddingScatter: 'Embedding Scatter',
  residuals: 'Residuals',
};

export const VISUALS: Record<VisualId, React.FC<VisualProps>> = {
  neuralNetForward: (props) => React.createElement(NeuralNetViz, { ...props, mode: 'forward' }),
  neuralNetBackprop: (props) => React.createElement(NeuralNetViz, { ...props, mode: 'backprop' }),
  gradDescent: GradientDescentViz,
  evaluation: EvaluationViz,
  confusionMatrix: ConfusionMatrixViz,
  embeddingScatter: EmbeddingScatterViz,
  residuals: ResidualsViz,
};
