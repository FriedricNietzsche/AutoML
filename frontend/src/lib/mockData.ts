import type { FileSystemNode } from './types';

const now = Date.now();

export const INITIAL_PIPELINE_CONFIG = {
  nodes: [
    { id: 'data_import', label: 'Data Import', status: 'pending', progress: 0, logs: [] },
    { id: 'validation', label: 'Data Validation', status: 'pending', progress: 0, logs: [] },
    { id: 'eda', label: 'EDA', status: 'pending', progress: 0, logs: [] },
    { id: 'preprocessing', label: 'Preprocessing', status: 'pending', progress: 0, logs: [] },
    { id: 'feature_eng', label: 'Feature Engineering', status: 'pending', progress: 0, logs: [] },
    { id: 'training', label: 'Model Training', status: 'pending', progress: 0, logs: [] },
    { id: 'evaluation', label: 'Evaluation', status: 'pending', progress: 0, logs: [] },
    { id: 'export', label: 'Export Model', status: 'pending', progress: 0, logs: [] },
  ]
};

const DEFAULT_METRICS = {
  accuracy: 0.0,
  f1_score: 0.0,
  auc_roc: 0.0,
  precision: 0.0,
  recall: 0.0,
  loss_history: []
};

const DEFAULT_CONFUSION_MATRIX = [
  [0, 0],
  [0, 0]
];

const TRAIN_PY = `import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_model():
    print("Initializing model...")
    model = Classifier(input_size=20, hidden_size=64, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training simulation logic happens in pipeline runner
    print("Model ready for training.")

if __name__ == "__main__":
    train_model()
`;

export const initialFileSystem: FileSystemNode[] = [
  {
    id: 'root',
    name: 'Files',
    type: 'folder',
    path: '/',
    isOpen: true,
    updatedAt: now,
    children: [
      {
        id: 'config',
        name: 'config',
        type: 'folder',
        path: '/config',
        isOpen: true,
        updatedAt: now,
        children: [
          {
            id: 'pipeline_json',
            name: 'pipeline.json',
            type: 'file',
            path: '/config/pipeline.json',
            content: JSON.stringify(INITIAL_PIPELINE_CONFIG, null, 2),
            updatedAt: now
          },
          {
            id: 'dataset_json',
            name: 'dataset.json',
            type: 'file',
            path: '/config/dataset.json',
            content: JSON.stringify({
              source: "s3://bucket/customer_churn_v2.csv",
              target_column: "churn",
              split: { train: 0.7, val: 0.15, test: 0.15 },
              features: ["age", "tenure", "balance", "products", "credit_score"]
            }, null, 2),
            updatedAt: now
          },
          {
            id: 'model_json',
            name: 'model.json',
            type: 'file',
            path: '/config/model.json',
            content: JSON.stringify({
              algorithm: "RandomForestClassifier",
              hyperparameters: {
                n_estimators: 100,
                max_depth: 10,
                min_samples_split: 2,
                random_state: 42
              }
            }, null, 2),
            updatedAt: now
          }
        ]
      },
      {
        id: 'src',
        name: 'src',
        type: 'folder',
        path: '/src',
        isOpen: true,
        updatedAt: now,
        children: [
          {
            id: 'train_py',
            name: 'train.py',
            type: 'file',
            path: '/src/train.py',
            content: TRAIN_PY,
            updatedAt: now
          },
          {
            id: 'utils_py',
            name: 'utils.py',
            type: 'file',
            path: '/src/utils.py',
            content: "def calculate_accuracy(y_true, y_pred):\n    return (y_true == y_pred).float().mean()",
            updatedAt: now
          }
        ]
      },
      {
        id: 'artifacts',
        name: 'artifacts',
        type: 'folder',
        path: '/artifacts',
        isOpen: true,
        updatedAt: now,
        children: [
          {
            id: 'metrics_json',
            name: 'metrics.json',
            type: 'file',
            path: '/artifacts/metrics.json',
            content: JSON.stringify(DEFAULT_METRICS, null, 2),
            updatedAt: now
          },
          {
            id: 'confusion_matrix_json',
            name: 'confusion_matrix.json',
            type: 'file',
            path: '/artifacts/confusion_matrix.json',
            content: JSON.stringify(DEFAULT_CONFUSION_MATRIX, null, 2),
            updatedAt: now
          },
          {
            id: 'training_log',
            name: 'training_log.txt',
            type: 'file',
            path: '/artifacts/training_log.txt',
            content: "Waiting for run to start...",
            updatedAt: now
          }
        ]
      },
      {
        id: 'preview_json',
        name: 'preview.json',
        type: 'file',
        path: '/preview.json',
        content: JSON.stringify({
          title: "AI Churn Predictor Dashboard",
          layout: "dashboard",
          active_pipeline: "churn_prediction_v1",
          views: ["pipeline", "metrics", "logs"]
        }, null, 2),
        updatedAt: now
      },
      {
        id: 'package_json',
        name: 'package.json',
        type: 'file',
        path: '/package.json',
        content: JSON.stringify({
          name: "ai-churn-predictor",
          version: "1.0.0",
          dependencies: {
            "torch": "^2.1.0",
            "pandas": "^2.1.0"
          }
        }, null, 2),
        updatedAt: now
      },
      {
        id: 'readme_md',
        name: 'README.md',
        type: 'file',
        path: '/README.md',
        content: "# AI Churn Predictor\n\n- Configure pipeline in `/config/pipeline.json`\n- Adjust model in `/config/model.json`\n- Run training to see artifacts in `/artifacts`",
        updatedAt: now
      }
    ]
  }
];

export const fileContent: Record<string, string> = {}; // Helper for legacy support if needed

export const mockFileTree = initialFileSystem;
export type { FileSystemNode as FileTreeNode } from './types';
