"""
Notebook Generator Module (Task 6.1)

Generates reproducible Jupyter notebooks from trained models.
Creates comprehensive notebooks showing the complete ML pipeline.

Features:
- nbformat-based notebook creation
- Complete pipeline reproduction
- Code + markdown cells
- Includes data loading, preprocessing, training, evaluation
- Inline visualizations
- Inference examples

Usage:
    generator = NotebookGenerator()
    notebook_path = generator.generate_notebook(
        project_id="my_project",
        run_id="run_abc123"
    )
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from .model_registry import ModelRegistry


# ============================================================================
# Notebook Generator
# ============================================================================

class NotebookGenerator:
    """
    Generate reproducible Jupyter notebooks from trained models.
    
    Creates notebooks with:
    - Project header and metadata
    - Environment setup
    - Data loading
    - Preprocessing pipeline
    - Model training
    - Evaluation and metrics
    - Visualizations
    - Inference examples
    - Results summary
    """
    
    def __init__(
        self,
        base_dir: str = "data/projects",
        emit_event: Optional[Callable] = None
    ):
        """
        Initialize NotebookGenerator.
        
        Args:
            base_dir: Base directory for project storage
            emit_event: Optional event emitter function
        """
        self.base_dir = Path(base_dir)
        self.emit_event = emit_event or (lambda *args, **kwargs: None)
        self.registry = ModelRegistry(base_dir=str(base_dir))
    
    def generate_notebook(
        self,
        project_id: str,
        run_id: str,
        output_path: Optional[str] = None,
        include_data_loading: bool = True,
        include_preprocessing: bool = True,
        include_evaluation: bool = True,
        include_inference: bool = True
    ) -> str:
        """
        Generate a Jupyter notebook for a specific training run.
        
        Args:
            project_id: Project identifier
            run_id: Training run identifier
            output_path: Optional custom output path
            include_data_loading: Include data loading cell
            include_preprocessing: Include preprocessing cells
            include_evaluation: Include evaluation cells
            include_inference: Include inference example
        
        Returns:
            Path to generated notebook
        """
        # Load metadata
        metadata = self.registry.get_metadata(run_id, project_id)
        if not metadata:
            raise ValueError(f"No metadata found for run {run_id}")
        
        # Create notebook
        notebook = new_notebook()
        
        # Add cells in order
        notebook.cells.extend(self._create_header_cells(metadata))
        notebook.cells.extend(self._create_setup_cells(metadata))
        
        if include_data_loading:
            notebook.cells.extend(self._create_data_loading_cells(metadata))
        
        if include_preprocessing:
            notebook.cells.extend(self._create_preprocessing_cells(metadata))
        
        notebook.cells.extend(self._create_training_cells(metadata))
        
        if include_evaluation:
            notebook.cells.extend(self._create_evaluation_cells(metadata))
        
        if include_inference:
            notebook.cells.extend(self._create_inference_cells(metadata))
        
        notebook.cells.extend(self._create_summary_cells(metadata))
        
        # Determine output path
        if output_path is None:
            run_dir = self.base_dir / project_id / "runs" / run_id
            output_path = str(run_dir / "notebook.ipynb")
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        
        # Emit event
        self._emit_notebook_ready_event(
            project_id=project_id,
            run_id=run_id,
            notebook_path=output_path,
            metadata=metadata
        )
        
        return output_path
    
    # ========================================================================
    # Cell Creation Methods
    # ========================================================================
    
    def _create_header_cells(self, metadata) -> List:
        """Create header and introduction cells."""
        cells = []
        
        # Title
        title = f"# {metadata.model_family} {metadata.task_type.capitalize()} Model"
        cells.append(new_markdown_cell(title))
        
        # Metadata info
        info_md = f"""
## Model Information

- **Run ID:** `{metadata.run_id}`
- **Model:** {metadata.model_family}
- **Task Type:** {metadata.task_type}
- **Trained:** {metadata.timestamp}
- **Primary Metric:** {metadata.primary_metric_name} = {metadata.primary_metric_value:.4f}

---

This notebook reproduces the complete training pipeline for this model.
All hyperparameters and configurations are preserved from the original training run.
"""
        cells.append(new_markdown_cell(info_md))
        
        return cells
    
    def _create_setup_cells(self, metadata) -> List:
        """Create environment setup cells."""
        cells = []
        
        # Setup header
        cells.append(new_markdown_cell("## Environment Setup"))
        
        # Imports
        imports_code = """# Import required libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import xgboost as xgb

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline
"""
        cells.append(new_code_cell(imports_code))
        
        # Version info
        version_md = f"""
### Environment Versions

- Python: `{metadata.python_version or 'Not recorded'}`
- XGBoost: `{metadata.xgboost_version or 'Not recorded'}`
- scikit-learn: `{metadata.sklearn_version or 'Not recorded'}`
"""
        cells.append(new_markdown_cell(version_md))
        
        return cells
    
    def _create_data_loading_cells(self, metadata) -> List:
        """Create data loading cells."""
        cells = []
        
        cells.append(new_markdown_cell("## Data Loading"))
        
        # Data loading code
        data_code = f"""# Load your dataset
# Replace this with your actual data loading code
# Example:
# df = pd.read_csv('your_data.csv')

# For this reproduction, you'll need to provide the dataset
# that was used during training.

# Dataset info from training:
# - Training samples: {metadata.n_train_samples}
# - Validation samples: {metadata.n_val_samples}
# - Test samples: {metadata.n_test_samples}
# - Features: {metadata.n_features}

print("Please load your dataset into a DataFrame called 'df'")
print(f"Expected features: {metadata.feature_names[:5]}..." if len(metadata.feature_names) > 5 else f"Expected features: {metadata.feature_names}")
"""
        cells.append(new_code_cell(data_code))
        
        return cells
    
    def _create_preprocessing_cells(self, metadata) -> List:
        """Create preprocessing cells."""
        cells = []
        
        cells.append(new_markdown_cell("## Data Preprocessing"))
        
        # Get preprocessing info from training config
        training_config = metadata.training_config
        
        preprocess_code = f"""# Split configuration from training
test_size = {training_config.get('test_size', 0.2)}
val_size = {training_config.get('val_size', 0.1)}
random_seed = {training_config.get('random_seed', 42)}

# Target column
target_column = '{training_config.get('target_column', 'target')}'

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed, stratify=y
)

# Train-validation split
val_ratio = val_size / (1 - test_size)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio, random_state=random_seed, stratify=y_temp
)

print(f"Training set: {{len(X_train)}} samples")
print(f"Validation set: {{len(X_val)}} samples")
print(f"Test set: {{len(X_test)}} samples")
"""
        cells.append(new_code_cell(preprocess_code))
        
        # Preprocessing note
        preprocess_note = """
**Note:** The original training used a preprocessing pipeline (encoding, scaling, etc.).
For exact reproduction, you would load the saved preprocessor:

```python
preprocessor = joblib.load('preprocessor.joblib')
X_train_transformed = preprocessor.transform(X_train)
X_val_transformed = preprocessor.transform(X_val)
X_test_transformed = preprocessor.transform(X_test)
```

For simplicity, this notebook assumes the data is already preprocessed or uses raw features.
"""
        cells.append(new_markdown_cell(preprocess_note))
        
        return cells
    
    def _create_training_cells(self, metadata) -> List:
        """Create model training cells."""
        cells = []
        
        cells.append(new_markdown_cell("## Model Training"))
        
        # Hyperparameters
        hyperparams_md = f"""
### Hyperparameters

The model was trained with the following configuration:

```json
{json.dumps(metadata.hyperparameters, indent=2)}
```
"""
        cells.append(new_markdown_cell(hyperparams_md))
        
        # Training code
        if metadata.model_family == "XGBoost":
            training_code = self._generate_xgboost_training_code(metadata)
        else:
            training_code = self._generate_generic_training_code(metadata)
        
        cells.append(new_code_cell(training_code))
        
        return cells
    
    def _generate_xgboost_training_code(self, metadata) -> str:
        """Generate XGBoost-specific training code."""
        params = metadata.hyperparameters
        
        # Determine objective
        if metadata.task_type == "classification":
            n_classes = params.get('num_class', 2)
            if n_classes == 2:
                objective = 'binary:logistic'
            else:
                objective = 'multi:softprob'
        else:
            objective = 'reg:squarederror'
        
        code = f"""# Initialize XGBoost model
model = xgb.XGBClassifier(
    n_estimators={params.get('n_estimators', 100)},
    max_depth={params.get('max_depth', 6)},
    learning_rate={params.get('learning_rate', 0.1)},
    subsample={params.get('subsample', 1.0)},
    colsample_bytree={params.get('colsample_bytree', 1.0)},
    objective='{objective}',
    random_state={params.get('random_state', 42)},
    n_jobs={params.get('n_jobs', -1)}
)

# Train the model
print("Training XGBoost model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Training complete!")
print(f"Best iteration: {{model.best_iteration}}")
"""
        return code
    
    def _generate_generic_training_code(self, metadata) -> str:
        """Generate generic training code."""
        code = f"""# Load the trained model
# You can load the saved model using:
model = joblib.load('{metadata.run_id}/model.joblib')

# Or retrain with the hyperparameters above
print("Model ready for evaluation")
"""
        return code
    
    def _create_evaluation_cells(self, metadata) -> List:
        """Create evaluation cells."""
        cells = []
        
        cells.append(new_markdown_cell("## Model Evaluation"))
        
        # Predictions
        prediction_code = """# Make predictions on test set
y_pred = model.predict(X_test)

# Get probabilities (for classification)
if hasattr(model, 'predict_proba'):
    y_pred_proba = model.predict_proba(X_test)
"""
        cells.append(new_code_cell(prediction_code))
        
        # Metrics
        if metadata.task_type == "classification":
            metrics_code = self._generate_classification_metrics_code(metadata)
        else:
            metrics_code = self._generate_regression_metrics_code(metadata)
        
        cells.append(new_code_cell(metrics_code))
        
        # Visualizations
        if metadata.task_type == "classification":
            viz_cells = self._create_classification_viz_cells(metadata)
            cells.extend(viz_cells)
        
        # Feature importance
        cells.extend(self._create_feature_importance_cells(metadata))
        
        return cells
    
    def _generate_classification_metrics_code(self, metadata) -> str:
        """Generate classification metrics code."""
        metrics = metadata.metrics
        
        code = f"""# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("="*50)
print("CLASSIFICATION METRICS")
print("="*50)
print(f"Accuracy:  {{accuracy:.4f}}")
print(f"Precision: {{precision:.4f}}")
print(f"Recall:    {{recall:.4f}}")
print(f"F1 Score:  {{f1:.4f}}")
print("="*50)

# Original training metrics (for comparison):
# Accuracy:  {metrics.get('accuracy', 'N/A')}
# Precision: {metrics.get('precision', 'N/A')}
# Recall:    {metrics.get('recall', 'N/A')}
# F1 Score:  {metrics.get('f1', 'N/A')}

# Detailed classification report
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
"""
        return code
    
    def _generate_regression_metrics_code(self, metadata) -> str:
        """Generate regression metrics code."""
        code = """# Calculate regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("="*50)
print("REGRESSION METRICS")
print("="*50)
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print("="*50)
"""
        return code
    
    def _create_classification_viz_cells(self, metadata) -> List:
        """Create classification visualization cells."""
        cells = []
        
        cells.append(new_markdown_cell("### Confusion Matrix"))
        
        cm_code = """# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.show()
"""
        cells.append(new_code_cell(cm_code))
        
        return cells
    
    def _create_feature_importance_cells(self, metadata) -> List:
        """Create feature importance visualization cells."""
        cells = []
        
        cells.append(new_markdown_cell("### Feature Importance"))
        
        importance_code = f"""# Plot feature importance (if available)
if hasattr(model, 'feature_importances_'):
    feature_names = {metadata.feature_names[:20]}  # Top 20 features
    importances = model.feature_importances_[:len(feature_names)]
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({{
        'feature': feature_names,
        'importance': importances
    }}).sort_values('importance', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['feature'], importance_df['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Top Features)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print("Feature importance not available for this model type")
"""
        cells.append(new_code_cell(importance_code))
        
        return cells
    
    def _create_inference_cells(self, metadata) -> List:
        """Create inference example cells."""
        cells = []
        
        cells.append(new_markdown_cell("## Inference Example"))
        
        inference_code = """# Example: Making predictions on new data
# Prepare a sample from the test set
sample_idx = 0
sample = X_test.iloc[[sample_idx]]

# Make prediction
prediction = model.predict(sample)[0]
print(f"Sample features:")
print(sample.T)
print(f"\\nPrediction: {prediction}")
print(f"Actual: {y_test.iloc[sample_idx]}")

# Show probability distribution (for classification)
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(sample)[0]
    print(f"\\nProbability distribution:")
    for i, prob in enumerate(probabilities):
        print(f"  Class {i}: {prob:.4f}")
"""
        cells.append(new_code_cell(inference_code))
        
        return cells
    
    def _create_summary_cells(self, metadata) -> List:
        """Create summary and conclusion cells."""
        cells = []
        
        cells.append(new_markdown_cell("## Summary"))
        
        summary_md = f"""
### Results Summary

This notebook reproduced the **{metadata.model_family}** model training pipeline.

**Key Results:**
- **Primary Metric ({metadata.primary_metric_name}):** {metadata.primary_metric_value:.4f}
- **Training Duration:** {metadata.training_duration_seconds:.2f} seconds
- **Samples:** {metadata.n_train_samples} train / {metadata.n_val_samples} val / {metadata.n_test_samples} test
- **Features:** {metadata.n_features}

**Model Configuration:**
```json
{json.dumps(metadata.hyperparameters, indent=2)}
```

### Next Steps

1. **Deploy:** Export this model for production use
2. **Optimize:** Try different hyperparameters to improve performance
3. **Analyze:** Investigate misclassifications or prediction errors
4. **Monitor:** Track model performance over time

---

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Run ID:** `{metadata.run_id}`
"""
        cells.append(new_markdown_cell(summary_md))
        
        return cells
    
    # ========================================================================
    # Event Emission
    # ========================================================================
    
    def _emit_notebook_ready_event(
        self,
        project_id: str,
        run_id: str,
        notebook_path: str,
        metadata
    ):
        """Emit NOTEBOOK_READY event."""
        # Get file size
        file_size = os.path.getsize(notebook_path)
        size_kb = file_size / 1024
        
        self.emit_event(
            "NOTEBOOK_READY",
            {
                "project_id": project_id,
                "run_id": run_id,
                "notebook_path": notebook_path,
                "size_kb": round(size_kb, 2),
                "model_family": metadata.model_family,
                "task_type": metadata.task_type,
                "primary_metric": {
                    "name": metadata.primary_metric_name,
                    "value": metadata.primary_metric_value
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def generate_comparison_notebook(
        self,
        project_id: str,
        run_ids: List[str],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a notebook comparing multiple models.
        
        Args:
            project_id: Project identifier
            run_ids: List of run IDs to compare
            output_path: Optional custom output path
        
        Returns:
            Path to generated comparison notebook
        """
        if len(run_ids) < 2:
            raise ValueError("Need at least 2 runs for comparison")
        
        # Load all metadata
        metadatas = []
        for run_id in run_ids:
            metadata = self.registry.get_metadata(run_id, project_id)
            if metadata:
                metadatas.append(metadata)
        
        if len(metadatas) < 2:
            raise ValueError("Could not load metadata for comparison")
        
        # Create notebook
        notebook = new_notebook()
        
        # Header
        notebook.cells.append(new_markdown_cell("# Model Comparison Report"))
        
        # Comparison table
        comparison_md = self._create_comparison_table(metadatas)
        notebook.cells.append(new_markdown_cell(comparison_md))
        
        # Metric comparison code
        comparison_code = self._create_comparison_code(metadatas)
        notebook.cells.append(new_code_cell(comparison_code))
        
        # Determine output path
        if output_path is None:
            project_dir = self.base_dir / project_id
            output_path = str(project_dir / "comparison_notebook.ipynb")
        
        # Write notebook
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook, f)
        
        return output_path
    
    def _create_comparison_table(self, metadatas: List) -> str:
        """Create markdown table comparing models."""
        md = "\n## Model Comparison\n\n"
        md += "| Run ID | Model | Metric | Value | Training Time |\n"
        md += "|--------|-------|--------|-------|---------------|\n"
        
        for m in metadatas:
            md += f"| {m.run_id[:8]}... | {m.model_family} | {m.primary_metric_name} | {m.primary_metric_value:.4f} | {m.training_duration_seconds:.2f}s |\n"
        
        return md
    
    def _create_comparison_code(self, metadatas: List) -> str:
        """Create code for visual comparison."""
        code = """import matplotlib.pyplot as plt
import numpy as np

# Model comparison data
models = """
        
        models_data = [
            {
                "name": f"{m.model_family} ({m.run_id[:8]})",
                "metric": m.primary_metric_value,
                "time": m.training_duration_seconds
            }
            for m in metadatas
        ]
        
        code += f"{models_data}\n\n"
        code += """# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Metric comparison
names = [m['name'] for m in models]
metrics = [m['metric'] for m in models]
ax1.bar(names, metrics)
ax1.set_ylabel('Metric Value')
ax1.set_title('Model Performance Comparison')
ax1.tick_params(axis='x', rotation=45)

# Training time comparison
times = [m['time'] for m in models]
ax2.bar(names, times, color='orange')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Training Time Comparison')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Find best model
best_idx = np.argmax(metrics)
print(f"\\nBest Model: {names[best_idx]}")
print(f"Metric: {metrics[best_idx]:.4f}")
"""
        return code


# ============================================================================
# Utility Functions
# ============================================================================

def generate_notebook_from_run(
    project_id: str,
    run_id: str,
    output_path: Optional[str] = None,
    emit_event: Optional[Callable] = None
) -> str:
    """
    Convenience function to generate a notebook from a training run.
    
    Args:
        project_id: Project identifier
        run_id: Training run identifier
        output_path: Optional custom output path
        emit_event: Optional event emitter function
    
    Returns:
        Path to generated notebook
    """
    generator = NotebookGenerator(emit_event=emit_event)
    return generator.generate_notebook(project_id, run_id, output_path)
