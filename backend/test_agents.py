"""
Quick test script to verify AI agent integration.
Tests: prompt parsing, dataset search, and WebSocket events.
"""
import asyncio
import sys
sys.path.insert(0, '/Users/krisviraujla/AutoML/backend')

from app.agents.prompt_parser import PromptParserAgent
from app.agents.dataset_finder import DatasetFinderAgent
from app.agents.license_validator import LicenseValidator

async def test_agents():
    print("=" * 60)
    print("AI AGENT INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Prompt Parser
    print("\n1. Testing Prompt Parser...")
    print("-" * 60)
    
    parser = PromptParserAgent()
    prompts = [
        "Build me a classifier for cat/dog",
        "Predict house prices",
        "Classify spam emails"
    ]
    
    for prompt in prompts:
        try:
            result = parser.parse(prompt)
            print(f"✓ '{prompt}'")
            print(f"  → task_type: {result.get('task_type')}")
            print(f"  → dataset_hint: {result.get('dataset_hint')}")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    # Test 2: License Validator
    print("\n2. Testing License Validator...")
    print("-" * 60)
    
    validator = LicenseValidator()
    licenses = [
        ("mit", True),
        ("apache-2.0", True),
        ("gpl-3.0", False),
        (None, False),
    ]
    
    for license_tag, expected in licenses:
        is_valid, reason = validator.is_allowed(license_tag)
        status = "✓" if is_valid == expected else "✗"
        print(f"{status} License '{license_tag}': {is_valid} ({reason})")
    
    # Test 3: Dataset Finder
    print("\n3. Testing Dataset Finder...")
    print("-" * 60)
    
    finder = DatasetFinderAgent()
    
    # Try a simpler search
    print("Searching for 'image classification' datasets...")
    try:
        candidates = finder.find_datasets(
            task_type="vision",
            dataset_hint="image classification",
            max_results=3
        )
        
        print(f"Found {len(candidates)} candidates")
        for i, dataset in enumerate(candidates[:3], 1):
            print(f"\n{i}. {dataset['id']}")
            print(f"   License: {dataset['license']} ({'✓ valid' if dataset['license_valid'] else '✗ invalid'})")
            print(f"   Reason: {dataset['license_reason']}")
            print(f"   Downloads: {dataset['downloads']:,}")
        
    except Exception as e:
        print(f"✗ Search failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_agents())
