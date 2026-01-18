#!/usr/bin/env python3
"""
Quick test script to verify the demo endpoint and WebSocket events.
"""
import asyncio
import sys
sys.path.insert(0, '/Users/krisviraujla/AutoML/backend')

from app.api.demo import run_demo_workflow

async def test_demo():
    """Test the demo workflow."""
    project_id = "test_demo"
    prompt = "Build me a classifier for cat/dog"
    
    print(f"Testing demo workflow for project: {project_id}")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    try:
        await run_demo_workflow(project_id, prompt)
        print("\n✅ Demo workflow completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("AutoML Demo Workflow Test")
    print("=" * 60)
    asyncio.run(test_demo())
