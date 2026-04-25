#!/usr/bin/env python3
"""
test_roboflow.py
Run this to verify your Roboflow credentials and dataset are accessible.
"""

import sys
from dotenv import load_dotenv
import os

load_dotenv()

RF_API_KEY   = os.getenv("ROBOFLOW_API_KEY")
RF_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
RF_PROJECT   = os.getenv("ROBOFLOW_PROJECT")
RF_VERSION   = os.getenv("ROBOFLOW_VERSION", "latest")

print("=== Roboflow Connection Test ===\n")

# Step 1: Check .env values are set
print("1. Checking .env values...")
missing = []
for name, val in [("ROBOFLOW_API_KEY", RF_API_KEY), ("ROBOFLOW_WORKSPACE", RF_WORKSPACE), ("ROBOFLOW_PROJECT", RF_PROJECT)]:
    if not val or val.startswith("your_"):
        missing.append(name)
if missing:
    print(f"   ❌  Missing values: {', '.join(missing)}")
    sys.exit(1)
print("   ✅  All values present\n")

# Step 2: Connect to Roboflow
print("2. Connecting to Roboflow...")
try:
    from roboflow import Roboflow
    rf = Roboflow(api_key=RF_API_KEY)
    print("   ✅  Connected\n")
except Exception as e:
    print(f"   ❌  Failed to connect: {e}")
    sys.exit(1)

# Step 3: Access workspace and project
print(f"3. Looking up project '{RF_PROJECT}' in workspace '{RF_WORKSPACE}'...")
try:
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    print(f"   ✅  Found project: {project.name}\n")
except Exception as e:
    print(f"   ❌  Could not find project: {e}")
    sys.exit(1)

# Step 4: Get version info
print("4. Checking dataset versions...")
try:
    versions = project.versions()
    if not versions:
        print("   ⚠️  No versions found — have you generated a dataset version in Roboflow?")
        sys.exit(1)
    print(f"   ✅  Found {len(versions)} version(s):")
    for v in versions:
        print(f"       - Version {v.version}: {v.splits} images")
    latest = versions[-1].version
    print(f"\n   Latest version: {latest}")
except Exception as e:
    print(f"   ❌  Could not retrieve versions: {e}")
    sys.exit(1)

print("\n=== All checks passed! Roboflow is ready. ===")