#!/usr/bin/env python3
"""
Test HuggingFace dataset search to debug why searches return 0 results
"""
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()

def test_hf_search():
    print("=" * 80)
    print("HuggingFace Dataset Search Test")
    print("=" * 80)
    
    # Initialize API
    hf_token = os.getenv("HF_TOKEN")
    print(f"\n1. Initializing HfApi...")
    print(f"   Token present: {bool(hf_token)}")
    print(f"   Token (first 20 chars): {hf_token[:20]}..." if hf_token else "   No token")
    
    if hf_token:
        hf_api = HfApi(token=hf_token)
    else:
        hf_api = HfApi()
    
    print("\n2. Testing basic list (no search, limit=10)...")
    try:
        count = 0
        for ds in hf_api.list_datasets(limit=10):
            count += 1
            print(f"   {count}. {ds.id}")
        print(f"   ✅ Retrieved {count} datasets")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n3. Testing search with 'titanic' (limit=10)...")
    try:
        count = 0
        for ds in hf_api.list_datasets(search="titanic", limit=10):
            count += 1
            print(f"   {count}. {ds.id}")
        print(f"   ✅ Found {count} datasets")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n4. Testing search with 'cats dogs' (limit=10)...")
    try:
        count = 0
        for ds in hf_api.list_datasets(search="cats dogs", limit=10):
            count += 1
            print(f"   {count}. {ds.id}")
        print(f"   ✅ Found {count} datasets")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n5. Testing search with 'image classification' (limit=10)...")
    try:
        count = 0
        for ds in hf_api.list_datasets(search="image classification", limit=10):
            count += 1
            print(f"   {count}. {ds.id}")
        print(f"   ✅ Found {count} datasets")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n6. Testing specific dataset lookup: 'cats_vs_dogs'...")
    try:
        count = 0
        for ds in hf_api.list_datasets(search="cats_vs_dogs", limit=10):
            count += 1
            print(f"   {count}. {ds.id}")
            print(f"       Downloads: {getattr(ds, 'downloads', 'N/A')}")
            print(f"       Tags: {getattr(ds, 'tags', [])}")
        print(f"   ✅ Found {count} datasets")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n7. Testing with author filter: 'author:huggingface'...")
    try:
        count = 0
        for ds in hf_api.list_datasets(author="huggingface", limit=10):
            count += 1
            print(f"   {count}. {ds.id}")
        print(f"   ✅ Found {count} datasets")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_hf_search()
