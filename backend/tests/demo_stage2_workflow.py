#!/usr/bin/env python3
"""
Stage 2 Workflow Demo: Complete Profiling Pipeline

Demonstrates the full Stage 2 data profiling workflow as it would be
integrated into the backend orchestrator.

Simulates:
  1. PROMPT_PARSED → extract task_type and target
  2. DATASET_SELECTED → load dataset
  3. PROFILE_PROGRESS → streaming updates
  4. PROFILE_SUMMARY → emit summary statistics
  5. MISSINGNESS_TABLE_READY → generate and serve JSON asset
  6. TARGET_DISTRIBUTION_READY → generate and serve PNG asset
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.ml.tabular.profiling import DataProfiler


def emit_event(event_name: str, payload: dict):
    """Simulate WebSocket event emission."""
    print(f"\n📡 WS EVENT: {event_name}")
    print(f"   Payload: {json.dumps(payload, indent=6)}")


def simulate_stage2_profiling(project_id: str = "proj_123"):
    """
    Simulate Stage 2 profiling workflow.
    
    This would be called by the backend orchestrator when:
    - User confirms Stage 1 (PROMPT_PARSED + MODEL_SELECTED)
    - User uploads dataset or selects demo dataset
    - Conductor transitions to Stage 2 (PROFILE_DATA)
    """
    print("=" * 80)
    print("STAGE 2: DATA PROFILING WORKFLOW")
    print("=" * 80)
    print()
    
    # ========================================================================
    # SETUP: Simulated context from Stage 1
    # ========================================================================
    stage1_context = {
        "task_type": "classification",
        "target": "churn",
        "dataset_hint": "customer churn prediction",
        "model_id": "random_forest_classifier"
    }
    
    print("✓ Stage 1 Complete:")
    print(f"   Task Type: {stage1_context['task_type']}")
    print(f"   Target: {stage1_context['target']}")
    print(f"   Selected Model: {stage1_context['model_id']}")
    print()
    
    # Simulate dataset (in real system, this comes from upload/selection)
    print("✓ Dataset Loaded (simulated customer churn data)")
    np.random.seed(42)
    n_rows = 5000
    
    df = pd.DataFrame({
        "customer_id": range(n_rows),
        "age": np.random.randint(18, 80, n_rows),
        "income": np.random.normal(60000, 25000, n_rows),
        "tenure_months": np.random.randint(1, 120, n_rows),
        "monthly_charges": np.random.normal(70, 30, n_rows),
        "total_spend": np.random.normal(2000, 1500, n_rows),
        "num_support_calls": np.random.randint(0, 10, n_rows),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n_rows),
        "payment_method": np.random.choice(["Credit card", "Bank transfer", "Electronic check"], n_rows),
        "has_phone_service": np.random.choice([0, 1], n_rows),
        "has_internet": np.random.choice([0, 1], n_rows),
        "churn": np.random.choice([0, 1], n_rows, p=[0.75, 0.25]),  # Imbalanced
    })
    
    # Introduce realistic data quality issues
    df.loc[np.random.choice(n_rows, int(n_rows * 0.12), replace=False), "income"] = np.nan
    df.loc[np.random.choice(n_rows, int(n_rows * 0.25), replace=False), "total_spend"] = np.nan
    df.loc[np.random.choice(n_rows, int(n_rows * 0.03), replace=False), "churn"] = np.nan
    
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print()
    
    # ========================================================================
    # PHASE 1: Initialize Profiler
    # ========================================================================
    print("-" * 80)
    print("PHASE 1: Initializing Profiler")
    print("-" * 80)
    
    emit_event("PROFILE_PROGRESS", {"phase": "initializing", "pct": 0})
    
    profiler = DataProfiler(
        df=df,
        target_column=stage1_context["target"],
        task_type=stage1_context["task_type"]
    )
    
    print("✓ DataProfiler initialized")
    time.sleep(0.5)  # Simulate work
    
    # ========================================================================
    # PHASE 2: Compute Profile Summary
    # ========================================================================
    print()
    print("-" * 80)
    print("PHASE 2: Computing Profile Summary")
    print("-" * 80)
    
    emit_event("PROFILE_PROGRESS", {"phase": "computing_summary", "pct": 20})
    
    summary = profiler.compute_profile()
    
    emit_event("PROFILE_SUMMARY", summary)
    
    print()
    print("✓ Profile Summary Computed:")
    print(f"   Rows: {summary['n_rows']:,}")
    print(f"   Columns: {summary['n_cols']}")
    print(f"   Missing: {summary['missing_pct']}%")
    print(f"   Types: {summary['types_breakdown']}")
    print(f"   Warnings: {len(summary['warnings'])} issue(s) detected")
    
    if summary['warnings']:
        print()
        print("   Data Quality Warnings:")
        for warning in summary['warnings']:
            print(f"      ⚠️  {warning}")
    
    time.sleep(1)  # Simulate work
    
    # ========================================================================
    # PHASE 3: Generate Missingness Table Asset
    # ========================================================================
    print()
    print("-" * 80)
    print("PHASE 3: Generating Missingness Table")
    print("-" * 80)
    
    emit_event("PROFILE_PROGRESS", {"phase": "generating_missingness", "pct": 50})
    
    asset_dir = backend_dir / "test_outputs" / f"project_{project_id}" / "profiling"
    missingness_path = profiler.generate_missingness_table(output_dir=str(asset_dir))
    
    # Simulate asset URL that frontend would fetch
    asset_url = f"/api/projects/{project_id}/assets/profiling/missingness_table.json"
    
    emit_event("MISSINGNESS_TABLE_READY", {"asset_url": asset_url})
    
    print()
    print(f"✓ Missingness Table Generated: {missingness_path}")
    
    # Show preview
    with open(missingness_path) as f:
        missingness_data = json.load(f)
    
    print()
    print("   Preview (top 5 columns by missing %):")
    for item in missingness_data[:5]:
        print(f"      {item['column']:20s} - {item['missing_pct']:5.1f}% missing ({item['missing_count']:,} / {item['missing_count'] + item['present_count']:,})")
    
    time.sleep(0.5)
    
    # ========================================================================
    # PHASE 4: Generate Target Distribution Plot
    # ========================================================================
    print()
    print("-" * 80)
    print("PHASE 4: Generating Target Distribution Plot")
    print("-" * 80)
    
    emit_event("PROFILE_PROGRESS", {"phase": "generating_distribution", "pct": 75})
    
    distribution_path = profiler.generate_target_distribution(output_dir=str(asset_dir))
    
    if distribution_path:
        asset_url = f"/api/projects/{project_id}/assets/profiling/target_distribution.png"
        
        emit_event("TARGET_DISTRIBUTION_READY", {"asset_url": asset_url})
        
        print()
        print(f"✓ Target Distribution Plot Generated: {distribution_path}")
        print(f"   Frontend can display: <img src=\"{asset_url}\" />")
    else:
        print()
        print("⚠️  Could not generate target distribution (matplotlib not available)")
    
    time.sleep(0.5)
    
    # ========================================================================
    # PHASE 5: Complete
    # ========================================================================
    print()
    print("-" * 80)
    print("PHASE 5: Profiling Complete")
    print("-" * 80)
    
    emit_event("PROFILE_PROGRESS", {"phase": "complete", "pct": 100})
    
    print()
    print("✓ Stage 2 Profiling Complete!")
    print()
    print("Next Steps:")
    print("   → Frontend displays ProfilingPanel with:")
    print("      - Summary cards (rows, cols, missing %, type breakdown)")
    print("      - Warnings panel (data quality issues)")
    print("      - Missingness table (interactive)")
    print("      - Target distribution chart")
    print()
    print("   → User reviews profile and clicks 'Confirm' to proceed to Stage 3 (Preprocessing)")
    print()
    
    # ========================================================================
    # SUMMARY STATS
    # ========================================================================
    print("=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)
    print()
    print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Target: {stage1_context['target']} ({stage1_context['task_type']})")
    print(f"Missing Data: {summary['missing_pct']}%")
    print(f"Data Quality Issues: {len(summary['warnings'])}")
    print()
    print("Assets Generated:")
    print(f"   1. {missingness_path}")
    if distribution_path:
        print(f"   2. {distribution_path}")
    print()
    print("WebSocket Events Emitted:")
    print("   1. PROFILE_PROGRESS (5 times)")
    print("   2. PROFILE_SUMMARY")
    print("   3. MISSINGNESS_TABLE_READY")
    if distribution_path:
        print("   4. TARGET_DISTRIBUTION_READY")
    print()


if __name__ == "__main__":
    try:
        simulate_stage2_profiling()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
