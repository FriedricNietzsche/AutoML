"""
Test Suite for Task 6.2: Export Bundle (ZIP)

Tests the enhanced ExportBundler with Jupyter notebook inclusion.

Run with:
    source .venv/bin/activate
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
    python backend/test_task62.py
"""

import asyncio
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from app.ml.tabular.export import ExportBundler
from app.ml.tabular.profiling import DataProfiler
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
        if event_type in ["EXPORT_READY", "NOTEBOOK_READY"]:
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

def test_export_with_notebook():
    """Test export bundle with auto-generated notebook."""
    return asyncio.run(async_test_export_with_notebook())


async def async_test_export_with_notebook():
    """Async test export bundle with notebook."""
    print("\n" + "="*70)
    print("TEST 1: Export Bundle with Notebook")
    print("="*70)
    
    collector = EventCollector()
    
    # Create and train a model
    print("\n1. Creating and training model...")
    df = load_iris(as_frame=True).frame
    df.columns = [c.replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    target_col = 'target'
    
    # Simple preprocessor
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f != target_col]
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_features)]
    )
    
    # Train model
    runner = TrainingRunner(
        emit_event=collector.emit,
        project_id="test_export_project"
    )
    
    result = await runner.run_training(
        df=df,
        target_column=target_col,
        task_type="classification",
        model_id="export_test_model",
        preprocessor=preprocessor,
        output_dir="data/projects/test_export_project/assets",
        n_estimators=30,
        max_depth=3,
        learning_rate=0.1
    )
    
    run_id = result['run_id']
    print(f"   ✅ Model trained: {run_id}")
    
    # Create export bundle
    print("\n2. Creating export bundle...")
    bundler = ExportBundler(emit_event=collector.emit)
    
    zip_path = bundler.create_bundle(
        run_id=run_id,
        project_id="test_export_project",
        include_source=True,
        include_notebook=True
    )
    
    print(f"   ✅ Bundle created: {zip_path}")
    
    # Verify ZIP exists
    print("\n3. Verifying bundle...")
    assert os.path.exists(zip_path), "ZIP file not created"
    
    # Check ZIP contents
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        file_list = zipf.namelist()
        print(f"   ✅ ZIP contains {len(file_list)} files")
        
        # Required files
        required_files = [
            'model.joblib',
            'metadata.json',
            'report.json',
            'notebook.ipynb',
            'README.md',
            'requirements.txt',
            'inference_example.py'
        ]
        
        for req_file in required_files:
            assert req_file in file_list, f"Missing required file: {req_file}"
            print(f"   ✅ Found: {req_file}")
        
        # Verify notebook content
        notebook_content = zipf.read('notebook.ipynb').decode('utf-8')
        assert len(notebook_content) > 0, "Notebook is empty"
        assert '"cells"' in notebook_content, "Notebook doesn't have cells"
        print(f"   ✅ Notebook is valid Jupyter notebook ({len(notebook_content)} bytes)")
        
        # Check README mentions notebook
        readme_content = zipf.read('README.md').decode('utf-8')
        assert 'notebook.ipynb' in readme_content, "README doesn't mention notebook"
        print(f"   ✅ README mentions notebook")
    
    # Get file size
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n   📦 Bundle size: {size_mb:.2f} MB")
    
    print("\n✅ TEST 1 PASSED: Export bundle with notebook works!")
    return zip_path, run_id


def test_export_ready_event():
    """Test EXPORT_READY event emission."""
    return asyncio.run(async_test_export_ready_event())


async def async_test_export_ready_event():
    """Async test EXPORT_READY event."""
    print("\n" + "="*70)
    print("TEST 2: EXPORT_READY Event")
    print("="*70)
    
    # Create export from test 1
    zip_path, run_id = await async_test_export_with_notebook()
    
    collector = EventCollector()
    
    # Emit event
    print("\n1. Emitting EXPORT_READY event...")
    bundler = ExportBundler()
    bundler.emit_export_ready_event(
        emit_fn=collector.emit,
        zip_path=zip_path,
        run_id=run_id
    )
    
    # Verify event
    export_events = collector.get_events_by_type("EXPORT_READY")
    assert len(export_events) > 0, "No EXPORT_READY event emitted"
    
    payload = export_events[0]["payload"]
    print(f"\n2. Verifying event payload...")
    print(f"   ✅ asset_url: {payload['asset_url']}")
    print(f"   ✅ filename: {payload['filename']}")
    print(f"   ✅ size_mb: {payload['size_mb']}")
    print(f"   ✅ checksum: {payload['checksum'][:16]}...")
    
    # Check contents list
    assert 'notebook.ipynb' in payload['contents'], "Notebook not in contents list"
    print(f"   ✅ Contents includes notebook.ipynb")
    
    print(f"\n   📋 Full contents: {', '.join(payload['contents'])}")
    
    print("\n✅ TEST 2 PASSED: EXPORT_READY event is correct!")


def test_export_without_notebook():
    """Test export bundle without notebook (optional)."""
    return asyncio.run(async_test_export_without_notebook())


async def async_test_export_without_notebook():
    """Async test export without notebook."""
    print("\n" + "="*70)
    print("TEST 3: Export Bundle WITHOUT Notebook")
    print("="*70)
    
    collector = EventCollector()
    
    # Create and train a model
    print("\n1. Training model...")
    df = load_iris(as_frame=True).frame
    df.columns = [c.replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f != 'target']
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_features)]
    )
    
    runner = TrainingRunner(
        emit_event=collector.emit,
        project_id="test_export_no_nb"
    )
    
    result = await runner.run_training(
        df=df,
        target_column='target',
        task_type="classification",
        model_id="no_notebook_model",
        preprocessor=preprocessor,
        output_dir="data/projects/test_export_no_nb/assets",
        n_estimators=20,
        max_depth=2,
        learning_rate=0.1
    )
    
    run_id = result['run_id']
    
    # Create export WITHOUT notebook
    print("\n2. Creating bundle without notebook...")
    bundler = ExportBundler()
    
    zip_path = bundler.create_bundle(
        run_id=run_id,
        project_id="test_export_no_nb",
        include_notebook=False  # Exclude notebook
    )
    
    # Verify notebook is NOT included
    print("\n3. Verifying notebook exclusion...")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        file_list = zipf.namelist()
        assert 'notebook.ipynb' not in file_list, "Notebook should not be included"
        print(f"   ✅ Notebook correctly excluded")
        print(f"   📋 Files: {', '.join(file_list)}")
    
    print("\n✅ TEST 3 PASSED: Export without notebook works!")


def test_bundle_contents_validity():
    """Test that all bundle contents are valid."""
    return asyncio.run(async_test_bundle_contents_validity())


async def async_test_bundle_contents_validity():
    """Async test bundle contents validity."""
    print("\n" + "="*70)
    print("TEST 4: Bundle Contents Validity")
    print("="*70)
    
    # Create export
    zip_path, _ = await async_test_export_with_notebook()
    
    print("\n1. Extracting and validating contents...")
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Validate JSON files
        for json_file in ['metadata.json', 'report.json']:
            content = zipf.read(json_file).decode('utf-8')
            data = json.loads(content)
            assert isinstance(data, dict), f"{json_file} is not a valid JSON object"
            print(f"   ✅ {json_file} is valid JSON")
        
        # Validate notebook is valid JSON (Jupyter format)
        notebook_content = zipf.read('notebook.ipynb').decode('utf-8')
        notebook_data = json.loads(notebook_content)
        assert 'cells' in notebook_data, "Notebook missing cells"
        assert 'metadata' in notebook_data, "Notebook missing metadata"
        assert len(notebook_data['cells']) > 0, "Notebook has no cells"
        print(f"   ✅ notebook.ipynb is valid Jupyter notebook")
        print(f"      - {len(notebook_data['cells'])} cells")
        
        # Validate README
        readme = zipf.read('README.md').decode('utf-8')
        assert len(readme) > 100, "README too short"
        assert '##' in readme, "README has no headers"
        print(f"   ✅ README.md is valid ({len(readme)} chars)")
        
        # Validate requirements.txt
        reqs = zipf.read('requirements.txt').decode('utf-8')
        assert 'xgboost' in reqs.lower() or 'scikit-learn' in reqs.lower(), \
            "requirements.txt missing ML libraries"
        print(f"   ✅ requirements.txt is valid")
        
        # Validate Python code
        inference_code = zipf.read('inference_example.py').decode('utf-8')
        assert 'import' in inference_code, "inference_example.py has no imports"
        assert 'def' in inference_code, "inference_example.py has no functions"
        print(f"   ✅ inference_example.py is valid Python")
    
    print("\n✅ TEST 4 PASSED: All bundle contents are valid!")


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
    print("Task 6.2: Export Bundle (ZIP) - Test Suite")
    print("🧪"*35)
    
    test_results = []
    
    # Test 1: Export with notebook
    try:
        test_export_with_notebook()
        test_results.append({"name": "Export with Notebook", "passed": True})
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        test_results.append({"name": "Export with Notebook", "passed": False})
        import traceback
        traceback.print_exc()
    
    # Test 2: EXPORT_READY event
    try:
        test_export_ready_event()
        test_results.append({"name": "EXPORT_READY Event", "passed": True})
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        test_results.append({"name": "EXPORT_READY Event", "passed": False})
        import traceback
        traceback.print_exc()
    
    # Test 3: Export without notebook
    try:
        test_export_without_notebook()
        test_results.append({"name": "Export without Notebook", "passed": True})
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        test_results.append({"name": "Export without Notebook", "passed": False})
        import traceback
        traceback.print_exc()
    
    # Test 4: Contents validity
    try:
        test_bundle_contents_validity()
        test_results.append({"name": "Bundle Contents Validity", "passed": True})
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        test_results.append({"name": "Bundle Contents Validity", "passed": False})
        import traceback
        traceback.print_exc()
    
    # Print summary
    print_summary(test_results)


if __name__ == "__main__":
    main()
