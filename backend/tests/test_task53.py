"""
Test Task 5.3: Model Registry, Leaderboard, and Export

Demonstrates:
1. Model persistence with metadata
2. Leaderboard tracking across multiple runs
3. Report generation
4. Export bundle creation
5. Model comparison
"""

import asyncio
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from app.ml.tabular.training import TrainingRunner
from app.ml.tabular.model_registry import ModelRegistry
from app.ml.tabular.leaderboard import LeaderboardManager
from app.ml.tabular.report_generator import ReportGenerator
from app.ml.tabular.export import ExportBundler


# ============================================================================
# Event Collector
# ============================================================================

class EventCollector:
    """Collects and displays events."""
    
    def __init__(self):
        self.events = []
        self.leaderboard_updates = []
        self.best_model_updates = []
    
    def emit(self, event_name: str, payload: dict):
        """Emit event handler."""
        self.events.append({"event": event_name, "payload": payload})
        
        if event_name == "LEADERBOARD_UPDATED":
            self.leaderboard_updates.append(payload)
            print(f"\n📊 LEADERBOARD UPDATED:")
            for entry in payload.get("entries", [])[:5]:
                print(f"  #{entry['rank']}: {entry['model']} - "
                      f"{payload['metric']}={entry['metric']:.4f}")
        
        elif event_name == "BEST_MODEL_UPDATED":
            self.best_model_updates.append(payload)
            print(f"\n🏆 NEW BEST MODEL:")
            print(f"  Run: {payload['run_id']}")
            print(f"  Metric: {payload['metric']['name']} = {payload['metric']['value']:.4f}")


# ============================================================================
# Test Functions
# ============================================================================

async def test_multiple_training_runs():
    """Test multiple training runs with leaderboard tracking."""
    print("\n" + "="*80)
    print("TEST: Multiple Training Runs with Leaderboard")
    print("="*80)
    
    # Create dataset
    data = {
        'Age': [22, 38, 26, 35, 35, 27, 54, 2, 27, 14] * 50,
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07, 11.13, 30.07] * 50,
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2] * 50,
        'Survived': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1] * 50,
    }
    df = pd.DataFrame(data)
    
    # Create preprocessor
    numeric_features = ['Age', 'Fare']
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_features)]
    )
    
    # Event collector
    collector = EventCollector()
    
    # Project ID
    project_id = "test_project_leaderboard"
    
    # Train 3 different models with different hyperparameters
    configs = [
        {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 40, "max_depth": 5, "learning_rate": 0.08},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n🚀 Training Model {i}/3...")
        print(f"   Config: {config}")
        
        runner = TrainingRunner(
            emit_event=collector.emit,
            project_id=project_id,
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        result = await runner.run_training(
            df=df,
            target_column='Survived',
            task_type='classification',
            model_id=f'xgboost_clf_v{i}',
            preprocessor=preprocessor,
            output_dir=f"assets/task53_test/run_{i}",
            **config
        )
        
        results.append(result)
        print(f"   ✓ Complete: f1={result['metrics']['metrics_dict']['f1']:.4f}")
    
    return results, collector, project_id


async def test_registry_operations(project_id: str):
    """Test model registry operations."""
    print("\n" + "="*80)
    print("TEST: Model Registry Operations")
    print("="*80)
    
    registry = ModelRegistry()
    
    # List all runs
    print("\n📁 All Runs:")
    runs = registry.list_runs(project_id)
    for run in runs:
        print(f"  - {run.run_id[:12]}: {run.model_family} "
              f"({run.primary_metric_name}={run.primary_metric_value:.4f})")
    
    # Get best model
    print("\n🏆 Best Model:")
    best = registry.get_best_model(project_id)
    if best:
        print(f"  Run ID: {best.run_id}")
        print(f"  Model: {best.model_family}")
        print(f"  Metric: {best.primary_metric_name} = {best.primary_metric_value:.4f}")
        print(f"  Hyperparameters: {best.hyperparameters}")
    
    # Compare models
    print("\n📊 Model Comparison:")
    run_ids = [run.run_id for run in runs[:3]]
    comparison = registry.compare_models(project_id, run_ids)
    
    print(f"  Comparing {len(comparison['runs'])} models:")
    for metric_name, values in comparison['metrics_comparison'].items():
        if metric_name == 'confusion_matrix':  # Skip non-scalar metrics
            continue
        print(f"\n  {metric_name}:")
        for item in values:
            if isinstance(item['value'], (int, float)):
                print(f"    {item['run_id'][:12]}: {item['value']:.4f}")
            else:
                print(f"    {item['run_id'][:12]}: {item['value']}")
    
    return registry


async def test_report_generation(project_id: str, run_id: str):
    """Test report generation."""
    print("\n" + "="*80)
    print("TEST: Report Generation")
    print("="*80)
    
    registry = ModelRegistry()
    report_gen = ReportGenerator()
    
    # Load metadata
    metadata = registry.get_metadata(run_id, project_id)
    
    # Generate summary text
    summary = report_gen.generate_summary_text(metadata)
    print("\n" + summary)
    
    # Load report JSON
    report = report_gen.load_report(run_id, project_id)
    print(f"\n📄 Report Generated:")
    print(f"  Version: {report['report_version']}")
    print(f"  Metrics: {len(report['metrics']['all_metrics'])} total")
    print(f"  Artifacts: {len(report['artifacts']['plots'])} plots")
    print(f"  Recommendations: {len(report['recommendations'])}")
    
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"\n  Recommendation {i}:")
        print(f"    {rec}")
    
    return report


async def test_export_bundle(project_id: str, run_id: str):
    """Test export bundle creation."""
    print("\n" + "="*80)
    print("TEST: Export Bundle Creation")
    print("="*80)
    
    bundler = ExportBundler()
    
    # Create bundle
    print("\n📦 Creating export bundle...")
    zip_path = bundler.create_bundle(
        run_id=run_id,
        project_id=project_id,
        include_source=True
    )
    
    print(f"  ✓ Bundle created: {zip_path}")
    
    # Get file size
    size_mb = Path(zip_path).stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    
    # List contents
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        files = zf.namelist()
        print(f"\n  Contents ({len(files)} files):")
        for file in sorted(files):
            info = zf.getinfo(file)
            file_size = info.file_size / 1024
            print(f"    - {file} ({file_size:.1f} KB)")
    
    return zip_path


async def test_leaderboard_stats(project_id: str):
    """Test leaderboard statistics."""
    print("\n" + "="*80)
    print("TEST: Leaderboard Statistics")
    print("="*80)
    
    leaderboard = LeaderboardManager()
    
    stats = leaderboard.get_statistics(project_id)
    
    print(f"\n📈 Leaderboard Stats:")
    print(f"  Total Runs: {stats['total_runs']}")
    print(f"  Best Score: {stats['best_score']:.4f}")
    print(f"  Worst Score: {stats['worst_score']:.4f}")
    print(f"  Average Score: {stats['avg_score']:.4f}")
    print(f"  Model Families: {', '.join(stats['model_families'])}")
    print(f"  Primary Metric: {stats['primary_metric']}")
    
    return stats


async def test_model_loading(project_id: str, run_id: str):
    """Test loading and using a saved model."""
    print("\n" + "="*80)
    print("TEST: Model Loading and Inference")
    print("="*80)
    
    registry = ModelRegistry()
    
    # Load model
    print(f"\n🔄 Loading model {run_id[:12]}...")
    model = registry.load_model(run_id, project_id)
    
    print(f"  ✓ Model loaded: {type(model).__name__}")
    
    # Test prediction
    test_data = pd.DataFrame({
        'Age': [25, 40, 30],
        'Fare': [50, 100, 75]
    })
    
    print("\n🎯 Making predictions:")
    predictions = model.predict(test_data)
    
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {pred}")
    
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(test_data)
        print("\n📊 Probabilities:")
        for i, proba in enumerate(probas):
            print(f"  Sample {i+1}: {proba}")
    
    return model


# ============================================================================
# Main Test Runner
# ============================================================================

async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("🧪 TASK 5.3: Model Registry & Export - Comprehensive Tests")
    print("="*80)
    
    # Test 1: Multiple training runs
    results, collector, project_id = await test_multiple_training_runs()
    
    print(f"\n✓ Trained {len(results)} models")
    print(f"✓ Captured {len(collector.events)} events")
    print(f"✓ Leaderboard updated {len(collector.leaderboard_updates)} times")
    print(f"✓ Best model updated {len(collector.best_model_updates)} times")
    
    # Get best run_id
    best_run_id = results[0]["run_id"]  # From first result
    
    # Test 2: Registry operations
    registry = await test_registry_operations(project_id)
    
    # Test 3: Report generation
    report = await test_report_generation(project_id, best_run_id)
    
    # Test 4: Leaderboard stats
    stats = await test_leaderboard_stats(project_id)
    
    # Test 5: Model loading
    model = await test_model_loading(project_id, best_run_id)
    
    # Test 6: Export bundle
    zip_path = await test_export_bundle(project_id, best_run_id)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE")
    print("="*80)
    
    print("\nKey Achievements:")
    print(f"  ✅ Trained {len(results)} models with different hyperparameters")
    print(f"  ✅ Models persisted with metadata to registry")
    print(f"  ✅ Leaderboard tracks all runs and ranks them")
    print(f"  ✅ Comprehensive reports generated")
    print(f"  ✅ Export bundles created with documentation")
    print(f"  ✅ Models can be loaded and used for inference")
    
    print(f"\n📁 Data saved to:")
    print(f"  Registry: data/projects/{project_id}/runs/")
    print(f"  Leaderboard: data/projects/{project_id}/leaderboard.json")
    print(f"  Export: {zip_path}")
    
    print("\n🎉 Task 5.3 Implementation Complete!")


if __name__ == "__main__":
    asyncio.run(main())
