"""
Test Suite for Task 6.1: Notebook Generation

Tests the NotebookGenerator module for creating reproducible Jupyter notebooks.

Run with:
    source .venv/bin/activate
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
    python backend/test_task61.py
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from app.ml.tabular.profiling import DataProfiler
from app.ml.tabular.notebook_generator import NotebookGenerator, generate_notebook_from_run
from app.ml.tabular.training import TrainingRunner


# ============================================================================
# Event Collector
# ============================================================================

class EventCollector:
    """Collect events emitted during testing."""
    
    def __init__(self):
        self.events = []
    
    def emit(self, event_type: str, payload: dict):
        """Collect an event."""
        self.events.append({
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.now().isoformat()
        })
        print(f"📢 {event_type}")
    
    def get_events_by_type(self, event_type: str):
        """Get all events of a specific type."""
        return [e for e in self.events if e["type"] == event_type]
    
    def clear(self):
        """Clear all events."""
        self.events = []


# ============================================================================
# Test Functions
# ============================================================================

def test_basic_notebook_generation():
    """Test basic notebook generation from a trained model."""
    return asyncio.run(async_test_basic_notebook_generation())

async def async_test_basic_notebook_generation():
    """Async test basic notebook generation from a trained model."""
    print("\n" + "="*70)
    print("TEST 1: Basic Notebook Generation")
    print("="*70)
    
    collector = EventCollector()
    
    # Create test dataset
    print("\n1. Creating test dataset...")
    df = load_iris(as_frame=True).frame
    df.columns = [c.replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    target_col = 'target'
    
    print(f"   Dataset: {len(df)} samples, {len(df.columns)} columns")
    
    # Profile data
    print("\n2. Profiling data...")
    profiler = DataProfiler(df, target_column=target_col, task_type="classification")
    profile = profiler.compute_profile()
    
    # Preprocess - create simple preprocessor
    print("\n3. Preprocessing...")
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f != target_col]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    # Train model
    print("\n4. Training model...")
    runner = TrainingRunner(
        emit_event=collector.emit,
        project_id="test_notebook_project"
    )
    
    result = await runner.run_training(
        df=df,
        target_column=target_col,
        task_type="classification",
        model_id="iris_xgboost",
        preprocessor=preprocessor,
        output_dir="data/projects/test_notebook_project/assets",
        n_estimators=30,
        max_depth=3,
        learning_rate=0.1
    )
    
    run_id = result['run_id']
    metrics = result['metrics']
    print(f"   Training complete: {metrics['primary_metric_name']} = {metrics['metrics_dict'][metrics['primary_metric_name'].replace('_', '')]:.4f}")
    
    # Generate notebook
    print("\n5. Generating notebook...")
    generator = NotebookGenerator(emit_event=collector.emit)
    
    notebook_path = generator.generate_notebook(
        project_id="test_notebook_project",
        run_id=run_id
    )
    
    print(f"   ✅ Notebook generated: {notebook_path}")
    
    # Verify notebook
    print("\n6. Verifying notebook...")
    assert os.path.exists(notebook_path), "Notebook file not created"
    
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    assert len(notebook.cells) > 0, "Notebook has no cells"
    print(f"   ✅ Notebook has {len(notebook.cells)} cells")
    
    # Check cell types
    markdown_cells = [c for c in notebook.cells if c.cell_type == 'markdown']
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    
    print(f"   ✅ Markdown cells: {len(markdown_cells)}")
    print(f"   ✅ Code cells: {len(code_cells)}")
    
    assert len(markdown_cells) > 0, "No markdown cells"
    assert len(code_cells) > 0, "No code cells"
    
    # Check for key sections
    notebook_text = '\n'.join([c.source for c in notebook.cells])
    
    required_sections = [
        "# XGBoost",  # Title
        "Environment Setup",
        "Data Loading",
        "Preprocessing",
        "Model Training",
        "Evaluation",
        "Inference",
        "Summary"
    ]
    
    for section in required_sections:
        assert section in notebook_text, f"Missing section: {section}"
        print(f"   ✅ Found section: {section}")
    
    # Check events
    notebook_events = collector.get_events_by_type("NOTEBOOK_READY")
    assert len(notebook_events) > 0, "No NOTEBOOK_READY event emitted"
    print(f"\n   ✅ NOTEBOOK_READY event emitted")
    
    event_payload = notebook_events[0]["payload"]
    print(f"   📊 Notebook size: {event_payload['size_kb']:.2f} KB")
    
    print("\n✅ TEST 1 PASSED: Basic notebook generation works!")
    return notebook_path, run_id


def test_notebook_content_quality():
    """Test the quality and completeness of notebook content."""
    print("\n" + "="*70)
    print("TEST 2: Notebook Content Quality")
    print("="*70)
    
    # Generate a notebook first
    print("\n1. Setting up test model...")
    notebook_path, run_id = test_basic_notebook_generation()
    
    # Read notebook
    print("\n2. Analyzing notebook content...")
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Check metadata presence
    print("\n3. Checking metadata...")
    notebook_text = '\n'.join([c.source for c in notebook.cells])
    
    metadata_items = [
        "Run ID",
        "Model:",
        "Task Type:",
        "Trained:",
        "Primary Metric:"
    ]
    
    for item in metadata_items:
        assert item in notebook_text, f"Missing metadata: {item}"
        print(f"   ✅ Found: {item}")
    
    # Check code quality
    print("\n4. Checking code cells...")
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    
    # Check for imports
    first_code = code_cells[0].source if code_cells else ""
    required_imports = ['pandas', 'numpy', 'joblib', 'xgboost', 'sklearn']
    
    for imp in required_imports:
        assert imp in first_code, f"Missing import: {imp}"
        print(f"   ✅ Import found: {imp}")
    
    # Check for hyperparameters
    assert 'n_estimators' in notebook_text, "Missing hyperparameters"
    assert 'max_depth' in notebook_text, "Missing hyperparameters"
    print(f"   ✅ Hyperparameters included")
    
    # Check for evaluation metrics
    eval_keywords = ['accuracy', 'precision', 'recall', 'f1']
    found_metrics = [kw for kw in eval_keywords if kw in notebook_text.lower()]
    assert len(found_metrics) > 0, "No evaluation metrics found"
    print(f"   ✅ Evaluation metrics: {', '.join(found_metrics)}")
    
    # Check for visualizations
    viz_keywords = ['confusion_matrix', 'feature_importance', 'plt.']
    found_viz = [kw for kw in viz_keywords if kw in notebook_text]
    assert len(found_viz) > 0, "No visualization code found"
    print(f"   ✅ Visualization code included")
    
    print("\n✅ TEST 2 PASSED: Notebook content is comprehensive!")


def test_comparison_notebook():
    """Test comparison notebook generation."""
    return asyncio.run(async_test_comparison_notebook())

async def async_test_comparison_notebook():
    """Async test comparison notebook generation."""
    print("\n" + "="*70)
    print("TEST 3: Comparison Notebook Generation")
    print("="*70)
    
    collector = EventCollector()
    
    # Create test dataset
    print("\n1. Creating test dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Profile and preprocess
    print("\n2. Profiling and preprocessing...")
    profiler = DataProfiler(df, target_column='target', task_type="classification")
    profile = profiler.compute_profile()
    
    # Create simple preprocessor
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f != 'target']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    # Train multiple models with different configs
    print("\n3. Training multiple models...")
    run_ids = []
    configs = [
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 75, "max_depth": 4, "learning_rate": 0.08}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n   Training model {i+1}/3 with config: {config}")
        runner = TrainingRunner(
            emit_event=collector.emit,
            project_id="test_comparison_project"
        )
        
        result = await runner.run_training(
            df=df,
            target_column='target',
            task_type="classification",
            model_id=f"model_v{i+1}",
            preprocessor=preprocessor,
            output_dir="data/projects/test_comparison_project/assets",
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate']
        )
        
        run_ids.append(result['run_id'])
        metrics = result['metrics']
        print(f"   ✅ Model {i+1}: {metrics['primary_metric_name']} = {metrics['metrics_dict'][metrics['primary_metric_name'].replace('_', '')]:.4f}")
    
    # Generate comparison notebook
    print("\n4. Generating comparison notebook...")
    generator = NotebookGenerator(emit_event=collector.emit)
    
    comparison_path = generator.generate_comparison_notebook(
        project_id="test_comparison_project",
        run_ids=run_ids
    )
    
    print(f"   ✅ Comparison notebook: {comparison_path}")
    
    # Verify comparison notebook
    print("\n5. Verifying comparison notebook...")
    assert os.path.exists(comparison_path), "Comparison notebook not created"
    
    with open(comparison_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    notebook_text = '\n'.join([c.source for c in notebook.cells])
    
    # Check for comparison elements
    assert "Model Comparison" in notebook_text, "Missing comparison header"
    assert len(run_ids) > 1, "Not enough models"
    
    # Check table
    assert "Run ID" in notebook_text, "Missing comparison table"
    assert "Model" in notebook_text, "Missing model column"
    assert "Metric" in notebook_text, "Missing metric column"
    
    print(f"   ✅ Comparison notebook has {len(notebook.cells)} cells")
    print(f"   ✅ Comparing {len(run_ids)} models")
    
    print("\n✅ TEST 3 PASSED: Comparison notebook generated!")
    return comparison_path


def test_convenience_function():
    """Test the convenience function."""
    print("\n" + "="*70)
    print("TEST 4: Convenience Function")
    print("="*70)
    
    # Use existing run from test 1
    _, run_id = test_basic_notebook_generation()
    
    print("\n1. Using convenience function...")
    collector = EventCollector()
    
    output_path = tempfile.mktemp(suffix='.ipynb')
    
    notebook_path = generate_notebook_from_run(
        project_id="test_notebook_project",
        run_id=run_id,
        output_path=output_path,
        emit_event=collector.emit
    )
    
    print(f"   ✅ Generated: {notebook_path}")
    
    # Verify
    assert os.path.exists(notebook_path), "Notebook not created"
    assert notebook_path == output_path, "Wrong output path"
    
    print("\n✅ TEST 4 PASSED: Convenience function works!")


def test_notebook_sections():
    """Test that all expected notebook sections are present."""
    print("\n" + "="*70)
    print("TEST 5: Notebook Section Validation")
    print("="*70)
    
    notebook_path, _ = test_basic_notebook_generation()
    
    print("\n1. Reading notebook...")
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    
    print("\n2. Validating sections...")
    
    # Expected sections in order
    expected_order = [
        "Model Information",
        "Environment Setup",
        "Data Loading",
        "Preprocessing",
        "Model Training",
        "Evaluation",
        "Inference",
        "Summary"
    ]
    
    notebook_text = '\n'.join([c.source for c in notebook.cells])
    
    last_pos = -1
    for section in expected_order:
        pos = notebook_text.find(section)
        assert pos > last_pos, f"Section '{section}' out of order or missing"
        last_pos = pos
        print(f"   ✅ {section} (position {pos})")
    
    print("\n✅ TEST 5 PASSED: All sections in correct order!")


def print_summary(test_results):
    """Print test summary."""
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    total = len(test_results)
    passed = sum(1 for r in test_results if r['passed'])
    
    for result in test_results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status}: {result['name']}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed")
    
    print("="*70)


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "🧪"*35)
    print("Task 6.1: Notebook Generation - Test Suite")
    print("🧪"*35)
    
    test_results = []
    
    # Test 1: Basic generation
    try:
        test_basic_notebook_generation()
        test_results.append({"name": "Basic Notebook Generation", "passed": True})
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        test_results.append({"name": "Basic Notebook Generation", "passed": False})
        import traceback
        traceback.print_exc()
    
    # Test 2: Content quality
    try:
        test_notebook_content_quality()
        test_results.append({"name": "Notebook Content Quality", "passed": True})
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        test_results.append({"name": "Notebook Content Quality", "passed": False})
        import traceback
        traceback.print_exc()
    
    # Test 3: Comparison notebook
    try:
        test_comparison_notebook()
        test_results.append({"name": "Comparison Notebook", "passed": True})
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        test_results.append({"name": "Comparison Notebook", "passed": False})
        import traceback
        traceback.print_exc()
    
    # Test 4: Convenience function
    try:
        test_convenience_function()
        test_results.append({"name": "Convenience Function", "passed": True})
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        test_results.append({"name": "Convenience Function", "passed": False})
        import traceback
        traceback.print_exc()
    
    # Test 5: Section validation
    try:
        test_notebook_sections()
        test_results.append({"name": "Section Validation", "passed": True})
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        test_results.append({"name": "Section Validation", "passed": False})
        import traceback
        traceback.print_exc()
    
    # Print summary
    print_summary(test_results)


if __name__ == "__main__":
    main()
