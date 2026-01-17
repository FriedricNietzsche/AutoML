#!/usr/bin/env python3
"""
Test script for DataProfiler (Task 4.1)

Tests the data profiling functionality with synthetic datasets.
Demonstrates:
  - Profile computation (PROFILE_SUMMARY)
  - Missingness table generation (MISSINGNESS_TABLE_READY)
  - Target distribution plot generation (TARGET_DISTRIBUTION_READY)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add backend to path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.ml.tabular.profiling import DataProfiler


def print_separator(char="="):
    print("\n" + char * 80 + "\n")


def print_event(event_name: str, payload: dict):
    """Pretty print a WebSocket event."""
    print(f"📡 EVENT: {event_name}")
    print(json.dumps(payload, indent=2))
    print_separator("-")


def create_classification_dataset(n_rows=1000):
    """Create a synthetic classification dataset with quality issues."""
    np.random.seed(42)
    
    data = {
        "customer_id": range(n_rows),  # High cardinality
        "age": np.random.randint(18, 80, n_rows),
        "income": np.random.normal(50000, 20000, n_rows),
        "credit_score": np.random.randint(300, 850, n_rows),
        "account_balance": np.random.normal(5000, 10000, n_rows),
        "num_products": np.random.randint(1, 5, n_rows),
        "is_active": np.random.choice([True, False], n_rows),
        "region": np.random.choice(["North", "South", "East", "West"], n_rows),
        "churn": np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),  # Imbalanced!
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    df.loc[np.random.choice(n_rows, int(n_rows * 0.15), replace=False), "age"] = np.nan
    df.loc[np.random.choice(n_rows, int(n_rows * 0.30), replace=False), "credit_score"] = np.nan
    df.loc[np.random.choice(n_rows, int(n_rows * 0.05), replace=False), "churn"] = np.nan
    
    return df


def create_regression_dataset(n_rows=800):
    """Create a synthetic regression dataset."""
    np.random.seed(123)
    
    data = {
        "property_id": range(n_rows),
        "sqft": np.random.randint(500, 5000, n_rows),
        "bedrooms": np.random.randint(1, 6, n_rows),
        "bathrooms": np.random.randint(1, 4, n_rows),
        "year_built": np.random.randint(1950, 2024, n_rows),
        "lot_size": np.random.randint(1000, 20000, n_rows),
        "neighborhood": np.random.choice(["Downtown", "Suburb", "Rural"], n_rows),
        "has_garage": np.random.choice([0, 1], n_rows),
        "price": np.random.normal(300000, 100000, n_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Add some outliers to price
    outlier_indices = np.random.choice(n_rows, 50, replace=False)
    df.loc[outlier_indices, "price"] = np.random.uniform(800000, 1500000, 50)
    
    # Add missing values
    df.loc[np.random.choice(n_rows, int(n_rows * 0.10), replace=False), "lot_size"] = np.nan
    df.loc[np.random.choice(n_rows, int(n_rows * 0.60), replace=False), "year_built"] = np.nan  # High missing!
    
    return df


def test_classification_profiling():
    """Test profiling a classification dataset."""
    print("=" * 80)
    print("TEST 1: Classification Dataset (Customer Churn)")
    print("=" * 80)
    print()
    
    # Create dataset
    df = create_classification_dataset(n_rows=1000)
    print(f"✓ Created dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Target: churn (classification)")
    print()
    
    # Initialize profiler
    profiler = DataProfiler(
        df=df,
        target_column="churn",
        task_type="classification"
    )
    print("✓ DataProfiler initialized")
    print_separator()
    
    # Compute profile summary
    print("Step 1: Computing profile summary...")
    summary = profiler.compute_profile()
    print_event("PROFILE_SUMMARY", summary)
    
    # Generate missingness table
    print("Step 2: Generating missingness table...")
    output_dir = backend_dir / "test_outputs" / "classification"
    missingness_path = profiler.generate_missingness_table(output_dir=str(output_dir))
    print(f"✓ Missingness table saved: {missingness_path}")
    
    # Load and display sample
    with open(missingness_path) as f:
        missingness_data = json.load(f)
    print("\nTop 3 columns by missing %:")
    for item in missingness_data[:3]:
        print(f"  - {item['column']}: {item['missing_pct']}% missing ({item['missing_count']}/{item['missing_count'] + item['present_count']})")
    
    print_event("MISSINGNESS_TABLE_READY", {"asset_url": f"/api/assets/{missingness_path}"})
    
    # Generate target distribution plot
    print("Step 3: Generating target distribution plot...")
    distribution_path = profiler.generate_target_distribution(output_dir=str(output_dir))
    if distribution_path:
        print(f"✓ Target distribution plot saved: {distribution_path}")
        print_event("TARGET_DISTRIBUTION_READY", {"asset_url": f"/api/assets/{distribution_path}"})
    else:
        print("⚠️  Could not generate target distribution plot")
    
    print_separator()


def test_regression_profiling():
    """Test profiling a regression dataset."""
    print("=" * 80)
    print("TEST 2: Regression Dataset (House Prices)")
    print("=" * 80)
    print()
    
    # Create dataset
    df = create_regression_dataset(n_rows=800)
    print(f"✓ Created dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Target: price (regression)")
    print()
    
    # Initialize profiler
    profiler = DataProfiler(
        df=df,
        target_column="price",
        task_type="regression"
    )
    print("✓ DataProfiler initialized")
    print_separator()
    
    # Compute profile summary
    print("Step 1: Computing profile summary...")
    summary = profiler.compute_profile()
    print_event("PROFILE_SUMMARY", summary)
    
    # Generate assets
    print("Step 2: Generating assets...")
    output_dir = backend_dir / "test_outputs" / "regression"
    
    missingness_path = profiler.generate_missingness_table(output_dir=str(output_dir))
    print(f"✓ Missingness table: {missingness_path}")
    
    distribution_path = profiler.generate_target_distribution(output_dir=str(output_dir))
    if distribution_path:
        print(f"✓ Target distribution: {distribution_path}")
    
    print_separator()


def test_profiling_without_target():
    """Test profiling a dataset without a target column."""
    print("=" * 80)
    print("TEST 3: Dataset Without Target (Clustering)")
    print("=" * 80)
    print()
    
    # Create dataset
    df = pd.DataFrame({
        "feature_1": np.random.randn(500),
        "feature_2": np.random.randn(500),
        "feature_3": np.random.choice(["A", "B", "C"], 500),
        "feature_4": np.random.randint(1, 100, 500),
    })
    
    print(f"✓ Created dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Target: None (clustering/unsupervised)")
    print()
    
    # Initialize profiler
    profiler = DataProfiler(df=df)
    print("✓ DataProfiler initialized")
    print_separator()
    
    # Compute profile
    summary = profiler.compute_profile()
    print_event("PROFILE_SUMMARY", summary)
    
    # Generate missingness table only (no target distribution)
    output_dir = backend_dir / "test_outputs" / "no_target"
    missingness_path = profiler.generate_missingness_table(output_dir=str(output_dir))
    print(f"✓ Missingness table: {missingness_path}")
    
    distribution_path = profiler.generate_target_distribution(output_dir=str(output_dir))
    if distribution_path:
        print(f"✓ Target distribution: {distribution_path}")
    else:
        print("ℹ️  No target distribution generated (no target column specified)")
    
    print_separator()


def main():
    """Run all profiling tests."""
    print("\n🧪 Testing DataProfiler (Task 4.1)\n")
    
    try:
        test_classification_profiling()
        test_regression_profiling()
        test_profiling_without_target()
        
        print("\n" + "=" * 80)
        print("✓ All profiling tests complete!")
        print("=" * 80)
        print("\nGenerated assets can be found in:")
        print(f"  {backend_dir / 'test_outputs'}/")
        print("\nCheck the following files:")
        print("  - classification/missingness_table.json")
        print("  - classification/target_distribution.png")
        print("  - regression/missingness_table.json")
        print("  - regression/target_distribution.png")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
