"""
Test Training Runner with Real XGBoost Streaming (Task 5.1)

Run this to see REAL streaming events from XGBoost callbacks!
Watch the console for progress updates after each boosting iteration.
"""

import asyncio
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from app.ml.tabular.training import TrainingRunner


# ============================================================================
# Event Collector - Captures emitted events
# ============================================================================

class EventCollector:
    """Collects and displays events in real-time."""
    
    def __init__(self):
        self.events = []
        self.progress_counter = 0
    
    def emit(self, event_name: str, payload: dict):
        """Emit event handler."""
        self.events.append({"event": event_name, "payload": payload})
        
        # Display important events
        if event_name == "TRAIN_RUN_STARTED":
            print(f"\n🚀 {event_name}")
            print(f"   Run ID: {payload['run_id']}")
            print(f"   Model: {payload['model_id']}")
            print(f"   Primary Metric: {payload['metric_primary']}")
            print(f"   Steps: {payload['config']['train']['steps']}\n")
        
        elif event_name == "TRAIN_PROGRESS":
            # Show every 10th progress update
            self.progress_counter += 1
            if self.progress_counter % 10 == 0 or payload['step'] == payload['steps']:
                eta_str = f"{payload['eta_s']:.1f}s" if payload['eta_s'] else "calculating..."
                print(f"⏳ Progress: {payload['step']}/{payload['steps']} | ETA: {eta_str}")
        
        elif event_name == "METRIC_SCALAR":
            # Show validation metrics only (less verbose)
            if payload['split'] == 'val' and payload['step'] % 20 == 0:
                print(f"   📊 {payload['name']} ({payload['split']}): {payload['value']:.4f}")
        
        elif event_name == "LOG_LINE":
            emoji = {"info": "ℹ️ ", "warning": "⚠️ ", "error": "❌"}
            print(f"{emoji.get(payload['level'], '')} {payload['text']}")
        
        elif event_name == "CONFUSION_MATRIX_READY":
            print(f"✅ Confusion matrix saved")
        
        elif event_name == "RESIDUALS_PLOT_READY":
            print(f"✅ Residuals plot saved")
        
        elif event_name == "FEATURE_IMPORTANCE_READY":
            print(f"✅ Feature importance saved")
        
        elif event_name == "TRAIN_RUN_FINISHED":
            print(f"\n🎉 {event_name}")
            print(f"   Status: {payload['status']}")
            if payload['status'] == 'success':
                metrics = payload['final_metrics']
                primary = metrics['primary']
                print(f"   Primary Metric: {primary['name']} = {primary['value']:.4f}")
                print(f"\n   All Metrics:")
                for m in metrics['metrics']:
                    print(f"      {m['name']}: {m['value']:.4f}")
    
    def summary(self):
        """Print event summary."""
        print(f"\n📋 Event Summary:")
        print(f"   Total events: {len(self.events)}")
        event_counts = {}
        for e in self.events:
            event_counts[e['event']] = event_counts.get(e['event'], 0) + 1
        for event, count in sorted(event_counts.items()):
            print(f"      {event}: {count}")


# ============================================================================
# Test Cases
# ============================================================================

async def test_classification_streaming():
    """Test REAL streaming with classification (Titanic dataset)."""
    print("\n" + "="*80)
    print("TEST 1: Classification with Real XGBoost Streaming (Titanic)")
    print("="*80)
    
    # Create synthetic Titanic-like data
    data = {
        'Age': [22, 38, 26, 35, 35, 27, 54, 2, 27, 14] * 50,
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07, 11.13, 30.07] * 50,
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2] * 50,
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'female'] * 50,
        'Embarked': ['S', 'C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'] * 50,
        'Survived': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1] * 50,
    }
    df = pd.DataFrame(data)
    
    # Create preprocessor
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create event collector
    collector = EventCollector()
    
    # Create training runner
    runner = TrainingRunner(
        emit_event=collector.emit,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Run training (50 estimators for visible streaming)
    output_dir = "assets/test_classification"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    result = await runner.run_training(
        df=df,
        target_column='Survived',
        task_type='classification',
        model_id='xgboost_classifier',
        preprocessor=preprocessor,
        output_dir=output_dir,
        n_estimators=50,  # 50 steps = 50 progress updates
        max_depth=4,
        learning_rate=0.1
    )
    
    # Show results
    print(f"\n📦 Artifacts saved to: {result['artifacts']}")
    collector.summary()
    
    return collector


async def test_regression_streaming():
    """Test REAL streaming with regression (Boston Housing)."""
    print("\n" + "="*80)
    print("TEST 2: Regression with Real XGBoost Streaming (House Prices)")
    print("="*80)
    
    # Create synthetic housing data
    data = {
        'RM': [6.575, 6.421, 7.185, 6.998, 7.147, 6.43, 6.012, 6.172, 5.631, 6.004] * 50,
        'LSTAT': [4.98, 9.14, 4.03, 2.94, 5.33, 5.21, 12.43, 19.15, 29.93, 17.10] * 50,
        'PTRATIO': [15.3, 17.8, 17.8, 18.7, 18.7, 15.2, 22.9, 18.9, 16.8, 20.2] * 50,
        'NOX': [0.469, 0.469, 0.524, 0.458, 0.524, 0.524, 0.871, 0.693, 0.573, 0.693] * 50,
        'MEDV': [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9] * 50,  # Target
    }
    df = pd.DataFrame(data)
    
    # Create preprocessor
    numeric_features = ['RM', 'LSTAT', 'PTRATIO', 'NOX']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
        ]
    )
    
    # Create event collector
    collector = EventCollector()
    
    # Create training runner
    runner = TrainingRunner(
        emit_event=collector.emit,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Run training
    output_dir = "assets/test_regression"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    result = await runner.run_training(
        df=df,
        target_column='MEDV',
        task_type='regression',
        model_id='xgboost_regressor',
        preprocessor=preprocessor,
        output_dir=output_dir,
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1
    )
    
    # Show results
    print(f"\n📦 Artifacts saved to: {result['artifacts']}")
    collector.summary()
    
    return collector


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("🔥 REAL STREAMING TRAINING WITH XGBOOST 🔥")
    print("="*80)
    print("\nWatch for:")
    print("  - TRAIN_RUN_STARTED with config")
    print("  - TRAIN_PROGRESS updates (every iteration)")
    print("  - METRIC_SCALAR with REAL values from XGBoost")
    print("  - RESOURCE_STATS (CPU/RAM usage)")
    print("  - Artifact events (plots saved)")
    print("  - TRAIN_RUN_FINISHED with final metrics")
    
    # Test classification
    collector1 = await test_classification_streaming()
    
    # Test regression
    collector2 = await test_regression_streaming()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE")
    print("="*80)
    print("\nKey Achievements:")
    print("  ✅ Real XGBoost callbacks (no simulation)")
    print("  ✅ Progress updates per boosting iteration")
    print("  ✅ Real metrics streamed during training")
    print("  ✅ Artifacts generated (confusion matrix, feature importance, etc.)")
    print("  ✅ Works for classification AND regression")
    
    # Save event log
    log_path = "assets/training_event_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            "classification_events": collector1.events,
            "regression_events": collector2.events
        }, f, indent=2, default=str)
    print(f"\n📄 Full event log saved to: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
