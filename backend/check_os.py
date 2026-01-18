#!/usr/bin/env python3
"""Check available Vultr OS images"""
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

api_key = os.getenv("VULTR_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print("Available OS Images for ML Training:")
print("=" * 80)

response = requests.get("https://api.vultr.com/v2/os", headers=headers)
if response.status_code == 200:
    os_list = response.json().get("os", [])
    
    # Filter for Ubuntu
    ubuntu_os = [os for os in os_list if "ubuntu" in os.get("name", "").lower()]
    
    print("\nUbuntu Options:")
    for os_img in ubuntu_os:
        print(f"  ID: {os_img['id']} - {os_img['name']}")
        print(f"     Arch: {os_img.get('arch', 'N/A')}")
        print(f"     Family: {os_img.get('family', 'N/A')}")
        print()
else:
    print(f"Failed: {response.status_code} - {response.text}")
