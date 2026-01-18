#!/usr/bin/env python3
"""Check available Vultr plans and regions for GPU instances"""
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

api_key = os.getenv("VULTR_API_KEY")
if not api_key:
    print("❌ VULTR_API_KEY not found in .env")
    exit(1)

base_url = "https://api.vultr.com/v2"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print("=" * 80)
print("Checking Vultr GPU Plans & Regions")
print("=" * 80)

# 1. Check available plans
print("\n[1/3] Available GPU Plans:")
print("-" * 80)

response = requests.get(f"{base_url}/plans", headers=headers)
if response.status_code == 200:
    plans = response.json().get("plans", [])
    gpu_plans = [p for p in plans if "gpu" in p.get("id", "").lower() or 
                 "vhp" in p.get("id", "").lower() or
                 p.get("gpu_vram", 0) > 0]
    
    if gpu_plans:
        for plan in gpu_plans[:10]:  # Show first 10
            print(f"  ✅ {plan['id']}")
            print(f"     Type: {plan.get('type', 'N/A')}")
            print(f"     vCPU: {plan.get('vcpu_count', 'N/A')}")
            print(f"     RAM: {plan.get('ram', 'N/A')} MB")
            print(f"     Disk: {plan.get('disk', 'N/A')} GB")
            print(f"     GPU VRAM: {plan.get('gpu_vram', 0)} GB")
            print(f"     Price: ${plan.get('monthly_cost', 'N/A')}/month")
            print(f"     Locations: {len(plan.get('locations', []))} regions")
            print()
    else:
        print("  ⚠️  No GPU plans found")
        print("  ℹ️  Showing regular compute plans instead:")
        for plan in plans[:5]:
            print(f"     {plan['id']} - {plan.get('vcpu_count')} vCPU, {plan.get('ram')} MB RAM")
else:
    print(f"  ❌ Failed to get plans: {response.status_code}")
    print(f"     {response.text}")

# 2. Check available regions
print("\n[2/3] Available Regions:")
print("-" * 80)

response = requests.get(f"{base_url}/regions", headers=headers)
if response.status_code == 200:
    regions = response.json().get("regions", [])
    for region in regions[:10]:  # Show first 10
        capabilities = region.get("options", [])
        has_gpu = any("gpu" in str(opt).lower() for opt in capabilities)
        
        print(f"  {'🎮' if has_gpu else '  '} {region['id']} - {region.get('city', 'N/A')}, {region.get('country', 'N/A')}")
        if has_gpu:
            print(f"     ✅ GPU Available")
else:
    print(f"  ❌ Failed to get regions: {response.status_code}")

# 3. Try to create a test instance with the current config
print("\n[3/3] Testing Instance Creation (DRY RUN):")
print("-" * 80)

test_config = {
    "region": "ewr",
    "plan": "vhp-1c-2gb-nvidia-a40-1",
    "label": "test-automl",
    "os_id": 387,
    "enable_ipv6": False,
    "backups": "disabled",
    "ddos_protection": False,
}

print(f"Config: {test_config}")
print("\nAttempting to validate (will fail if plan/region invalid)...")

response = requests.post(f"{base_url}/instances", headers=headers, json=test_config)
if response.status_code == 201:
    print("  ✅ Config is valid!")
    # Delete the test instance immediately
    instance_id = response.json()["instance"]["id"]
    requests.delete(f"{base_url}/instances/{instance_id}", headers=headers)
    print("  ℹ️  Test instance deleted")
elif response.status_code == 400:
    print(f"  ❌ Invalid configuration!")
    print(f"     Error: {response.text}")
    
    # Try to parse error
    try:
        error_data = response.json()
        if "error" in error_data:
            print(f"     Details: {error_data['error']}")
    except:
        pass
else:
    print(f"  ⚠️  Unexpected status: {response.status_code}")
    print(f"     {response.text}")

print("\n" + "=" * 80)
print("Done! Use the plan IDs shown above in your code.")
print("=" * 80)
