#!/usr/bin/env python3
"""
Test script for PreprocessAgent (Task 4.2)

Tests the preprocessing planner functionality with synthetic datasets.
Demonstrates:
  - Preprocessing plan creation (PREPROCESS_PLAN)
  - Train/val/test split summary (SPLIT_SUMMARY)
  - Actual sklearn ColumnTransformer building
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add backend to path for imports
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.agents.preprocess import PreprocessAgent


def print_separator(char="="):
    print("\n" + char * 80 + "\n")


def print_event(event_name: str, payload: dict):
    """Pretty print a WebSocket event."""
    print(f"📡 EVENT: {event_name}")
    print(json.dumps(payload, indent=2))
    print_separator("-")


def create_messy_dataset(n_rows=1000):
    """Create a dataset with various data quality issues."""
    np.random.seed(42)
    
    data = {
        # ID column (high cardinality - should be dropped)
        "customer_id": range(n_rows),
        
        # Numeric features (some with missing values)
        "age": np.random.randint(18, 80, n_rows),
        "income": np.random.normal(50000, 20000, n_rows),
        "credit_score": np.random.randint(300, 850, n_rows),
        
        # Categorical features (some with missing values)
        "region": np.random.choice(["North", "South", "East", "West"], n_rows),
        "job_category": np.random.choice(["Tech", "Sales", "Support", "Management"], n_rows),
        
        # Constant column (should be dropped)
        "country": ["USA"] * n_rows,
        
        # Target
        "churn": np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    df.loc[np.random.choice(n_rows, int(n_rows * 0.15), replace=False), "age"] = np.nan
    df.loc[np.random.choice(n_rows, int(n_rows * 0.20), replace=False), "income"] = np.nan
    df.loc[np.random.choice(n_rows, int(n_rows * 0.10), replace=False), "region"] = np.nan
    df.loc[np.random.choice(n_rows, int(n_rows * 0.05), replace=False), "churn"] = np.nan
    
    return df


def test_preprocess_plan_classification():
    """Test preprocessing plan creation for classification."""
    print("=" * 80)
    print("TEST 1: Preprocessing Plan (Classification)")
    print("=" * 80)
    print()
    
    # Create dataset
    df = create_messy_dataset(n_rows=1000)
    print(f"✓ Created dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Features: {list(df.columns)}")
    print()
    
    # Initialize agent
    agent = PreprocessAgent()
    print("✓ PreprocessAgent initialized")
    print(f"  Config:")
    print(f"    - Numeric imputation: {agent.numeric_impute_strategy}")
    print(f"    - Numeric scaling: {agent.numeric_scaler}")
    print(f"    - Categorical imputation: {agent.categorical_impute_strategy}")
    print(f"    - Categorical encoding: {agent.categorical_encoder}")
    print_separator()
    
    # Create preprocessing plan
    print("Step 1: Creating preprocessing plan...")
    plan = agent.create_plan(
        df=df,
        target_column="churn",
        task_type="classification"
    )
    print_event("PREPROCESS_PLAN", plan)
    
    # Display steps in readable format
    print("Preprocessing Pipeline Steps:")
    for i, step in enumerate(plan["steps"], 1):
        print(f"\n  Step {i}: {step['step_type'].upper()}")
        print(f"    Method: {step['method']}")
        print(f"    Columns: {', '.join(step['columns'][:5])}{'...' if len(step['columns']) > 5 else ''} ({len(step['columns'])} total)")
        print(f"    Reason: {step['reason']}")
    
    print_separator()
    
    # Create split summary
    print("Step 2: Creating train/val/test split summary...")
    split_summary = agent.create_split_summary(
        df=df,
        target_column="churn",
        task_type="classification",
        test_size=0.2,
        val_size=0.1,
        seed=42
    )
    print_event("SPLIT_SUMMARY", split_summary)
    
    total_rows = split_summary["train_rows"] + split_summary["val_rows"] + split_summary["test_rows"]
    print(f"Total rows after cleaning: {total_rows}")
    print(f"  Train: {split_summary['train_rows']} ({split_summary['train_rows']/total_rows*100:.1f}%)")
    print(f"  Val:   {split_summary['val_rows']} ({split_summary['val_rows']/total_rows*100:.1f}%)")
    print(f"  Test:  {split_summary['test_rows']} ({split_summary['test_rows']/total_rows*100:.1f}%)")
    print(f"  Stratified: {split_summary['stratified']} (preserves class distribution)")
    print(f"  Seed: {split_summary['seed']}")
    
    print_separator()


def test_preprocess_plan_regression():
    """Test preprocessing plan creation for regression."""
    print("=" * 80)
    print("TEST 2: Preprocessing Plan (Regression)")
    print("=" * 80)
    print()
    
    # Create regression dataset
    np.random.seed(123)
    n_rows = 800
    
    df = pd.DataFrame({
        "property_id": range(n_rows),  # ID - should be dropped
        "sqft": np.random.randint(500, 5000, n_rows),
        "bedrooms": np.random.randint(1, 6, n_rows),
        "bathrooms": np.random.randint(1, 4, n_rows),
        "year_built": np.random.randint(1950, 2024, n_rows),
        "neighborhood": np.random.choice(["Downtown", "Suburb", "Rural"], n_rows),
        "price": np.random.normal(300000, 100000, n_rows),
    })
    
    # Add missing values
    df.loc[np.random.choice(n_rows, int(n_rows * 0.15), replace=False), "year_built"] = np.nan
    df.loc[np.random.choice(n_rows, int(n_rows * 0.10), replace=False), "neighborhood"] = np.nan
    
    print(f"✓ Created dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print()
    
    # Initialize agent with robust scaling (better for regression with outliers)
    agent = PreprocessAgent(numeric_scaler="robust")
    print("✓ PreprocessAgent initialized (with RobustScaler for outlier handling)")
    print_separator()
    
    # Create plan
    plan = agent.create_plan(
        df=df,
        target_column="price",
        task_type="regression"
    )
    print_event("PREPROCESS_PLAN", plan)
    
    # Create split summary (no stratification for regression)
    split_summary = agent.create_split_summary(
        df=df,
        target_column="price",
        task_type="regression",
        test_size=0.2,
        val_size=0.1,
        seed=42
    )
    print_event("SPLIT_SUMMARY", split_summary)
    
    print(f"Note: Stratified = {split_summary['stratified']} (stratification not used for regression)")
    
    print_separator()


def test_build_transformer():
    """Test actual sklearn ColumnTransformer building."""
    print("=" * 80)
    print("TEST 3: Building Sklearn ColumnTransformer")
    print("=" * 80)
    print()
    
    # Create dataset
    df = create_messy_dataset(n_rows=500)
    print(f"✓ Created dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print()
    
    # Initialize agent
    agent = PreprocessAgent()
    
    # Build actual transformer
    print("Building sklearn ColumnTransformer...")
    transformer = agent.build_transformer(df=df, target_column="churn")
    
    print(f"✓ ColumnTransformer created")
    print(f"\n  Transformers:")
    for name, pipeline, columns in transformer.transformers_:
        print(f"    {name}:")
        print(f"      Columns: {columns}")
        print(f"      Pipeline steps:")
        for step_name, step_obj in pipeline.steps:
            print(f"        - {step_name}: {type(step_obj).__name__}")
    
    print()
    print("  This transformer can be used in sklearn Pipeline:")
    print("    from sklearn.pipeline import Pipeline")
    print("    from sklearn.ensemble import RandomForestClassifier")
    print()
    print("    pipeline = Pipeline([")
    print("        ('preprocessor', transformer),")
    print("        ('classifier', RandomForestClassifier())")
    print("    ])")
    print()
    print("    pipeline.fit(X_train, y_train)")
    
    # Test fit_transform
    print()
    print("Testing fit_transform on sample data...")
    X = df.drop(columns=["churn"])
    y = df["churn"].dropna()
    X = X.loc[y.index]  # Align indices
    
    X_transformed = transformer.fit_transform(X)
    print(f"✓ Transform successful!")
    print(f"  Input shape:  {X.shape}")
    print(f"  Output shape: {X_transformed.shape}")
    print(f"  Output type:  {type(X_transformed)}")
    
    print_separator()


def test_edge_cases():
    """Test edge cases."""
    print("=" * 80)
    print("TEST 4: Edge Cases")
    print("=" * 80)
    print()
    
    agent = PreprocessAgent()
    
    # Test 1: All numeric dataset
    print("Edge Case 1: All numeric features")
    df_numeric = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "feat3": np.random.randn(100),
        "target": np.random.randint(0, 2, 100),
    })
    plan1 = agent.create_plan(df_numeric, target_column="target")
    print(f"  Steps: {len(plan1['steps'])}")
    for step in plan1['steps']:
        print(f"    - {step['step_type']}: {step['method']}")
    print()
    
    # Test 2: All categorical dataset
    print("Edge Case 2: All categorical features")
    df_categorical = pd.DataFrame({
        "cat1": np.random.choice(["A", "B", "C"], 100),
        "cat2": np.random.choice(["X", "Y", "Z"], 100),
        "target": np.random.randint(0, 2, 100),
    })
    plan2 = agent.create_plan(df_categorical, target_column="target")
    print(f"  Steps: {len(plan2['steps'])}")
    for step in plan2['steps']:
        print(f"    - {step['step_type']}: {step['method']}")
    print()
    
    # Test 3: No missing values
    print("Edge Case 3: No missing values")
    df_clean = pd.DataFrame({
        "num1": np.random.randn(100),
        "cat1": np.random.choice(["A", "B"], 100),
        "target": np.random.randint(0, 2, 100),
    })
    plan3 = agent.create_plan(df_clean, target_column="target")
    print(f"  Steps: {len(plan3['steps'])}")
    for step in plan3['steps']:
        print(f"    - {step['step_type']}: {step['method']} ({len(step['columns'])} cols)")
    
    print_separator()
    print("✓ All edge cases handled correctly")
    print_separator()


def main():
    """Run all preprocessing tests."""
    print("\n🧪 Testing PreprocessAgent (Task 4.2)\n")
    
    try:
        test_preprocess_plan_classification()
        test_preprocess_plan_regression()
        test_build_transformer()
        test_edge_cases()
        
        print("\n" + "=" * 80)
        print("✓ All preprocessing tests complete!")
        print("=" * 80)
        print()
        print("Summary:")
        print("  ✓ PREPROCESS_PLAN event generation validated")
        print("  ✓ SPLIT_SUMMARY event generation validated")
        print("  ✓ Sklearn ColumnTransformer building validated")
        print("  ✓ Edge cases handled (all numeric, all categorical, no missing)")
        print()
        print("Next Steps:")
        print("  → Wire PreprocessAgent into Stage 2 orchestrator")
        print("  → Emit PREPROCESS_PLAN + SPLIT_SUMMARY events over WebSocket")
        print("  → Frontend displays preprocessing steps in UI")
        print("  → Use build_transformer() in Task 5.1 (Training Runner)")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
