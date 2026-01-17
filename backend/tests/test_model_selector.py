#!/usr/bin/env python3
"""
Test script for ModelSelectorAgent (Task 3.3)

Tests the model selection logic for different task types.
This is a rule-based agent, so no LLM/API keys needed.
"""

import json
import sys
from pathlib import Path

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.agents.model_selector import ModelSelectorAgent


def print_separator():
    print("\n" + "=" * 80 + "\n")


def test_model_selector():
    """Test ModelSelectorAgent with various task types."""
    
    print("🧪 Testing ModelSelectorAgent")
    print_separator()

    # Initialize agent
    agent = ModelSelectorAgent()
    print("✓ ModelSelectorAgent initialized")
    print_separator()

    # Test cases: different task types
    test_cases = [
        {
            "name": "Classification Task",
            "task_type": "classification",
            "constraints": {"metric": "F1 > 0.85"},
        },
        {
            "name": "Regression Task",
            "task_type": "regression",
            "constraints": {"training_time_limit": "10 minutes"},
        },
        {
            "name": "Clustering Task",
            "task_type": "clustering",
            "constraints": {},
        },
        {
            "name": "Time Series Task",
            "task_type": "timeseries",
            "constraints": {},
        },
        {
            "name": "NLP Task",
            "task_type": "nlp",
            "constraints": {"gpu_available": True},
        },
        {
            "name": "Vision Task",
            "task_type": "vision",
            "constraints": {},
        },
        {
            "name": "Unknown Task Type",
            "task_type": "unknown_type",
            "constraints": {},
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"  Task Type: {test_case['task_type']}")
        
        # Get model candidates
        candidates_payload = agent.get_candidates(
            task_type=test_case["task_type"],
            constraints=test_case["constraints"],
        )
        
        print(f"  Number of Candidates: {len(candidates_payload['models'])}")
        
        # Pretty print candidates
        for model in candidates_payload["models"]:
            print(f"\n  Model: {model['name']}")
            print(f"    ID: {model['id']}")
            print(f"    Family: {model['family']}")
            print(f"    Why: {model['why']}")
            print(f"    Requirements: {json.dumps(model['requirements'], indent=6)}")
        
        print_separator()

    # Test model selection
    print("Testing model selection (MODEL_SELECTED event):")
    print_separator()
    
    # Select a classification model
    selected = agent.select_model(model_id="random_forest_classifier")
    print(f"✓ Selected model: {selected['model_id']}")
    print(f"  Payload: {json.dumps(selected, indent=2)}")
    print_separator()

    # Test validation
    print("Testing model validation:")
    print_separator()
    
    validation_tests = [
        ("random_forest_classifier", "classification", True),
        ("linear_regression", "regression", True),
        ("random_forest_classifier", "regression", False),
        ("nonexistent_model", "classification", False),
    ]
    
    for model_id, task_type, expected in validation_tests:
        result = agent.validate_model_exists(model_id, task_type)
        status = "✓" if result == expected else "✗"
        print(f"{status} validate_model_exists('{model_id}', '{task_type}') = {result} (expected {expected})")
    
    print_separator()
    print("✓ All tests complete!")


if __name__ == "__main__":
    try:
        test_model_selector()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
