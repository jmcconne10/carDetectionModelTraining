#!/usr/bin/env python3
"""
check_instance.py
Shows how long your Vultr GPU instances have been running and estimated cost so far.
"""

import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

VULTR_API_KEY  = os.getenv("VULTR_API_KEY")
VULTR_API_BASE = "https://api.vultr.com/v2"
VULTR_HEADERS  = {"Authorization": f"Bearer {VULTR_API_KEY}"}

def format_duration(seconds):
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

r = requests.get(f"{VULTR_API_BASE}/instances", headers=VULTR_HEADERS)
instances = r.json().get("instances", [])

gpu_instances = [i for i in instances if i.get("plan", "").startswith("vcg")]

if not gpu_instances:
    print("No GPU instances currently running.")
else:
    print(f"\n{'Label':<30} {'IP':<18} {'Uptime':<18} {'Est. Cost'}")
    print("-" * 80)
    for i in gpu_instances:
        label      = i.get("label", i["id"])[:28]
        ip         = i.get("main_ip", "unknown")
        plan       = i.get("plan", "")
        created_at = i.get("date_created")
        status     = i.get("status")
        power      = i.get("power_status")

        # Calculate uptime
        if created_at:
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            uptime_secs = (datetime.now(timezone.utc) - created_dt).total_seconds()
            uptime_str  = format_duration(uptime_secs)
            uptime_hrs  = uptime_secs / 3600
        else:
            uptime_str = "unknown"
            uptime_hrs = 0

        # Look up hourly cost for the plan
        plans_r = requests.get(f"{VULTR_API_BASE}/plans?type=vcg", headers=VULTR_HEADERS)
        plans   = plans_r.json().get("plans", [])
        hourly  = next((p["hourly_cost"] for p in plans if p["id"] == plan), None)
        cost_str = f"${uptime_hrs * hourly:.4f}" if hourly else "unknown"

        print(f"{label:<30} {ip:<18} {uptime_str:<18} {cost_str}")

    print()