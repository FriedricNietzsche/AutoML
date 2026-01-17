"""
Export Bundle Module (Task 5.3 - Phase 4)

Creates downloadable export bundles (ZIP) with model, artifacts, and documentation.
Lazy generation - only creates ZIP when explicitly requested.

Bundle contents:
  - model.joblib (trained pipeline)
  - metadata.json (full model metadata)
  - report.json (comprehensive report)
  - *.png (all plots)
  - inference_example.py (usage example)
  - requirements.txt (dependencies)
  - README.md (documentation)
"""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .model_registry import ModelMetadata, ModelRegistry
from .report_generator import ReportGenerator


# ============================================================================
# Export Bundler
# ============================================================================

class ExportBundler:
    """
    Creates downloadable export bundles for trained models.
    
    Features:
    - Lazy ZIP creation (on-demand)
    - Includes all artifacts and documentation
    - Generates standalone inference example
    - Creates comprehensive README
    - Emits EXPORT_READY event
    
    Usage:
        bundler = ExportBundler()
        
        # Create bundle
        zip_path = bundler.create_bundle(
            run_id="abc123",
            project_id="project_1"
        )
        
        # Emit event
        bundler.emit_export_ready_event(emit_fn, zip_path)
    """
    
    def __init__(self, base_dir: str = "data/projects"):
        """
        Initialize export bundler.
        
        Args:
            base_dir: Base directory for project data
        """
        self.base_dir = Path(base_dir)
        self.registry = ModelRegistry(base_dir=str(base_dir))
        self.report_gen = ReportGenerator(base_dir=str(base_dir))
    
    def create_bundle(
        self,
        run_id: str,
        project_id: str,
        include_source: bool = True,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Create complete export bundle as ZIP file.
        
        Args:
            run_id: Run identifier
            project_id: Project identifier
            include_source: Include inference example source code
            output_dir: Custom output directory (optional)
            
        Returns:
            Path to created .zip file
        """
        # Load metadata
        metadata = self.registry.get_metadata(run_id, project_id)
        
        # Determine output directory
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.base_dir / project_id / "exports"
        
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Create ZIP filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        zip_name = f"{metadata.model_id}_{run_id[:8]}_{timestamp}.zip"
        zip_path = out_path / zip_name
        
        # Get run directory
        run_dir = self.base_dir / project_id / "runs" / run_id
        
        # Create temporary bundle directory
        temp_dir = out_path / f"temp_{run_id}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy all artifacts from run directory
            for item in run_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, temp_dir / item.name)
            
            # Generate README
            readme_path = temp_dir / "README.md"
            readme_content = self.generate_readme(metadata)
            readme_path.write_text(readme_content)
            
            # Generate requirements.txt
            req_path = temp_dir / "requirements.txt"
            req_content = self.generate_requirements(metadata)
            req_path.write_text(req_content)
            
            # Generate inference example
            if include_source:
                example_path = temp_dir / "inference_example.py"
                example_content = self.generate_inference_example(metadata)
                example_path.write_text(example_content)
            
            # Create ZIP
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in temp_dir.rglob('*'):
                    if file.is_file():
                        arcname = file.relative_to(temp_dir)
                        zipf.write(file, arcname)
            
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        return str(zip_path)
    
    def generate_readme(self, metadata: ModelMetadata) -> str:
        """
        Generate README.md for the export bundle.
        
        Args:
            metadata: ModelMetadata instance
            
        Returns:
            README content as string
        """
        readme = f"""# {metadata.model_id}

**Trained Model Export**

Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}  
Run ID: `{metadata.run_id}`  
Task Type: {metadata.task_type.capitalize()}  
Model Family: {metadata.model_family}

---

## 📊 Performance Summary

**Primary Metric:** {metadata.primary_metric_name} = **{metadata.primary_metric_value:.4f}**

### All Metrics:
"""
        
        # Add all metrics
        for metric_name, value in sorted(metadata.metrics.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                readme += f"- **{metric_name}**: {value:.4f}\n"
        
        readme += f"""
---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Load and Use Model
```python
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load('model.joblib')

# Prepare your data (example)
data = pd.DataFrame({{
    'feature1': [value1],
    'feature2': [value2],
    # ... add all features
}})

# Make predictions
predictions = model.predict(data)
print(predictions)
```

See `inference_example.py` for a complete working example.

---

## 📁 Bundle Contents

- **model.joblib**: Complete trained pipeline (includes preprocessing)
- **metadata.json**: Full model metadata and configuration
- **report.json**: Comprehensive training report
- **README.md**: This file
- **requirements.txt**: Python dependencies
- **inference_example.py**: Working code example
"""
        
        # Add plots if available
        plot_files = [
            k for k in metadata.artifact_paths.keys()
            if k not in ['model']
        ]
        
        if plot_files:
            readme += "\n### Visualizations:\n"
            for plot_name in plot_files:
                readme += f"- **{plot_name}.png**: {plot_name.replace('_', ' ').title()} visualization\n"
        
        readme += f"""
---

## ⚙️ Model Configuration

**Hyperparameters:**
```json
{json.dumps(metadata.hyperparameters, indent=2)}
```

**Training Configuration:**
```json
{json.dumps(metadata.training_config, indent=2)}
```

---

## 📈 Training Details

- **Training Duration**: {metadata.training_duration_seconds:.1f} seconds
- **Train Samples**: {metadata.n_train_samples}
- **Validation Samples**: {metadata.n_val_samples}
- **Test Samples**: {metadata.n_test_samples}
- **Total Features**: {metadata.n_features}

---

## 🔧 System Information

- **Python Version**: {metadata.python_version}
- **XGBoost Version**: {metadata.xgboost_version}
- **Scikit-learn Version**: {metadata.sklearn_version}

---

## 📝 Notes

{metadata.notes if metadata.notes else "No additional notes."}

---

## 📄 License

This model was trained using the AutoML Agentic Builder.

**Tags**: {", ".join(metadata.tags) if metadata.tags else "none"}

---

Generated by AutoML Agentic Builder  
Run ID: {metadata.run_id}
"""
        
        return readme
    
    def generate_requirements(self, metadata: ModelMetadata) -> str:
        """
        Generate requirements.txt with necessary dependencies.
        
        Args:
            metadata: ModelMetadata instance
            
        Returns:
            requirements.txt content
        """
        requirements = [
            f"scikit-learn=={metadata.sklearn_version}",
            f"xgboost=={metadata.xgboost_version}",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "joblib>=1.3.0"
        ]
        
        return "\n".join(requirements) + "\n"
    
    def generate_inference_example(self, metadata: ModelMetadata) -> str:
        """
        Generate standalone inference example script.
        
        Args:
            metadata: ModelMetadata instance
            
        Returns:
            Python code as string
        """
        # Create sample feature values
        sample_features = {}
        for i, feature_name in enumerate(metadata.feature_names[:10]):
            sample_features[feature_name] = f"<value_{i+1}>"
        
        example = f'''#!/usr/bin/env python3
"""
Inference Example for {metadata.model_id}

This script demonstrates how to use the trained model for predictions.
"""

import joblib
import pandas as pd


def load_model(model_path='model.joblib'):
    """Load the trained model pipeline."""
    return joblib.load(model_path)


def prepare_data(raw_data):
    """
    Prepare data for prediction.
    
    Args:
        raw_data: Dictionary with feature values
        
    Returns:
        pandas DataFrame ready for prediction
    """
    return pd.DataFrame([raw_data])


def predict(model, data):
    """
    Make predictions using the model.
    
    Args:
        model: Loaded model pipeline
        data: Prepared pandas DataFrame
        
    Returns:
        Predictions array
    """
    return model.predict(data)


def predict_proba(model, data):
    """
    Get prediction probabilities (classification only).
    
    Args:
        model: Loaded model pipeline
        data: Prepared pandas DataFrame
        
    Returns:
        Probability array or None if regression
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(data)
    return None


def main():
    """Main inference workflow."""
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Example input data
    # Replace these values with your actual feature values
    sample_data = {json.dumps(sample_features, indent=8)}
    
    # Prepare data
    print("Preparing data...")
    df = prepare_data(sample_data)
    
    # Make prediction
    print("Making prediction...")
    prediction = predict(model, df)
    
    print(f"\\nPrediction: {{prediction[0]}}")
    
    # Get probabilities (if classification)
    probabilities = predict_proba(model, df)
    if probabilities is not None:
        print(f"Probabilities: {{probabilities[0]}}")
    
    return prediction


if __name__ == "__main__":
    result = main()
'''
        
        return example
    
    def emit_export_ready_event(
        self,
        emit_fn: Callable[[str, Dict], None],
        zip_path: str,
        run_id: str
    ):
        """
        Emit EXPORT_READY event for frontend.
        
        Args:
            emit_fn: Event emission function
            zip_path: Path to created ZIP file
            run_id: Run identifier
        """
        import hashlib
        
        # Calculate checksum
        with open(zip_path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        
        # Get file size
        size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
        
        payload = {
            "asset_url": f"/api/downloads/{Path(zip_path).name}",
            "run_id": run_id,
            "filename": Path(zip_path).name,
            "size_mb": round(size_mb, 2),
            "checksum": checksum,
            "contents": [
                "model.joblib",
                "metadata.json",
                "report.json",
                "README.md",
                "requirements.txt",
                "inference_example.py",
                "*.png (plots)"
            ]
        }
        
        emit_fn("EXPORT_READY", payload)


# ============================================================================
# Legacy Functions (backward compatibility)
# ============================================================================

def export_model(model, preprocessing_pipeline, model_name, export_dir):
    """Legacy function for backward compatibility."""
    import joblib
    import os

    os.makedirs(export_dir, exist_ok=True)
    model_path = os.path.join(export_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    pipeline_path = os.path.join(export_dir, f"{model_name}_pipeline.joblib")
    joblib.dump(preprocessing_pipeline, pipeline_path)
    
    return model_path, pipeline_path


def export_notebook(notebook_content, export_dir, model_name):
    """Legacy function for backward compatibility."""
    import os

    os.makedirs(export_dir, exist_ok=True)
    notebook_path = os.path.join(export_dir, f"{model_name}.ipynb")
    
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    
    return notebook_path


def export_report(report_data, export_dir, model_name):
    """Legacy function for backward compatibility."""
    import os
    import json

    os.makedirs(export_dir, exist_ok=True)
    report_path = os.path.join(export_dir, f"{model_name}_report.json")
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f)
    
    return report_path