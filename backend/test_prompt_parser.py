#!/usr/bin/env python3
"""
Standalone test for PromptParserAgent.

Run from backend/:
    python3 test_prompt_parser.py

This script:
1. Loads GEMINI_API_KEY from backend/.env
2. Creates a PromptParserAgent
3. Parses a sample prompt
4. Prints the output (PROMPT_PARSED payload)
"""

import os
import sys
from pathlib import Path

# Add backend to path so we can import app.*
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load .env
from dotenv import load_dotenv
load_dotenv(backend_dir / ".env", override=True)

from app.agents.prompt_parser import PromptParserAgent


def test_parse():
    print("=" * 60)
    print("Testing PromptParserAgent")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "replace_me":
        print("❌ ERROR: GEMINI_API_KEY not set in backend/.env")
        print("   Set a valid Gemini API key and try again.")
        sys.exit(1)
    
    print(f"✓ GEMINI_API_KEY loaded: {api_key[:10]}...")
    
    # Create agent
    try:
        agent = PromptParserAgent()
        print("✓ PromptParserAgent initialized")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        sys.exit(1)
    
    # Test prompts
    test_prompts = [
        "Build a model to predict customer churn from tabular data. I need F1 > 0.85 and training under 10 minutes.",
        "Classify images of cats vs dogs using a CNN",
        "Predict house prices from Zillow data",
        "",  # Empty prompt (should ask for clarification)
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print("\n" + "-" * 60)
        print(f"Test {i}/4")
        print("-" * 60)
        print(f"INPUT PROMPT:\n  {repr(prompt)}\n")
        
        try:
            result = agent.parse(prompt)
            print("OUTPUT (PROMPT_PARSED payload):")
            for key, value in result.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"❌ Parse failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ All tests complete")
    print("=" * 60)


if __name__ == "__main__":
    test_parse()
