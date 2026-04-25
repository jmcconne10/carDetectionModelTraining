#!/usr/bin/env python3
"""
terminate_instance.py
Finds and destroys any running Vultr GPU instances.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

VULTR_API_KEY  = os.getenv("VULTR_API_KEY")
VULTR_API_BASE = "https://api.vultr.com/v2"
VULTR_HEADERS  = {"Authorization": f"Bearer {VULTR_API_KEY}", "Content-Type": "application/json"}

r = requests.get(f"{VULTR_API_BASE}/instances", headers=VULTR_HEADERS)
instances = r.json().get("instances", [])

gpu_instances = [i for i in instances if i.get("plan", "").startswith("vcg")]

if not gpu_instances:
    print("No GPU instances currently running. Nothing to terminate.")
else:
    print(f"\nFound {len(gpu_instances)} GPU instance(s):\n")
    for i, instance in enumerate(gpu_instances):
        print(f"  [{i+1}] {instance.get('label', instance['id'])}  |  {instance.get('main_ip')}  |  {instance['id']}")

    print("\nOptions:")
    print("  Enter a number to terminate that instance")
    if len(gpu_instances) > 1:
        print("  Enter 'all' to terminate all instances")
    print("  Enter 'q' to quit without terminating\n")

    choice = input("Your choice: ").strip().lower()

    to_terminate = []
    if choice == 'q':
        print("Cancelled. No instances terminated.")
    elif choice == 'all' and len(gpu_instances) > 1:
        to_terminate = gpu_instances
    elif choice.isdigit() and 1 <= int(choice) <= len(gpu_instances):
        to_terminate = [gpu_instances[int(choice) - 1]]
    else:
        print("Invalid choice. No instances terminated.")

    for instance in to_terminate:
        label       = instance.get("label", instance["id"])
        instance_id = instance["id"]
        print(f"Terminating {label} ({instance_id})...", end=" ")
        r = requests.delete(f"{VULTR_API_BASE}/instances/{instance_id}", headers=VULTR_HEADERS)
        if r.status_code in (200, 204):
            print("✅  Done. Billing stopped.")
        else:
            print(f"❌  Failed (status {r.status_code}). Try deleting manually in your Vultr dashboard.")