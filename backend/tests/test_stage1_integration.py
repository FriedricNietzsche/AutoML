#!/usr/bin/env python3
"""
Integration test: Prompt Parser → Model Selector

Demonstrates Stage 1 workflow:
1. Parse user prompt → PROMPT_PARSED
2. Get model candidates based on task_type → MODEL_CANDIDATES
3. Select a model → MODEL_SELECTED

This simulates the backend orchestrator flow for Stage 1.
"""

import json
import os
import sys
from pathlib import Path

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

from app.agents.prompt_parser import PromptParserAgent
from app.agents.model_selector import ModelSelectorAgent


def print_separator(char="="):
    print("\n" + char * 80 + "\n")


def print_event(event_name: str, payload: dict):
    """Pretty print a WebSocket event."""
    print(f"📡 EVENT: {event_name}")
    print(json.dumps(payload, indent=2))
    print_separator("-")


def stage1_workflow(user_prompt: str):
    """
    Simulate Stage 1 workflow:
    - Parse prompt
    - Get model candidates
    - Auto-select first candidate (or user would choose)
    """
    print(f"👤 USER PROMPT: {user_prompt}")
    print_separator()

    # Step 1: Parse prompt
    print("Step 1: Parse user prompt...")
    parser = PromptParserAgent()
    parsed = parser.parse(user_prompt)
    print_event("PROMPT_PARSED", parsed)

    # Step 2: Get model candidates based on task_type
    print("Step 2: Get model candidates...")
    selector = ModelSelectorAgent()
    candidates = selector.get_candidates(
        task_type=parsed["task_type"],
        constraints=parsed.get("constraints", {}),
    )
    print_event("MODEL_CANDIDATES", candidates)

    # Step 3: Simulate user selecting first candidate (or auto-select)
    if candidates["models"]:
        selected_model_id = candidates["models"][0]["id"]
        print(f"Step 3: Auto-selecting first candidate: {selected_model_id}...")
        selected = selector.select_model(model_id=selected_model_id)
        print_event("MODEL_SELECTED", selected)

        # Validation check
        is_valid = selector.validate_model_exists(
            selected_model_id, parsed["task_type"]
        )
        print(f"✓ Validation: model '{selected_model_id}' is valid for task '{parsed['task_type']}': {is_valid}")
    else:
        print("⚠️  No model candidates found")

    print_separator()


def main():
    """Run integration tests with different prompts."""
    
    # Check if Gemini API key is loaded
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"✓ GEMINI_API_KEY loaded: {api_key[:20]}...")
    else:
        print("⚠️  GEMINI_API_KEY not set - prompt parsing may fail")
    
    print_separator()

    # Test cases
    test_prompts = [
        "Build a customer churn prediction model with F1 > 0.85",
        "Predict house prices using Zillow data",
        "Cluster customer segments for marketing campaign",
        "Classify images of cats and dogs using CNN",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'#' * 80}")
        print(f"# TEST {i}/{len(test_prompts)}")
        print(f"{'#' * 80}\n")
        stage1_workflow(prompt)

    print("\n✓ All integration tests complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
