#!/usr/bin/env python3
"""
train_on_vultr.py

Run from your Mac:  python3 train_on_vultr.py

What this does:
  1. Looks up your latest Roboflow dataset version
  2. Spins up a Vultr GPU instance with a cloud-init startup script
  3. The instance automatically installs deps, downloads your dataset, and trains
  4. Polls via SSH until training finishes
  5. Downloads best.pt to your Mac
  6. Destroys the instance (so you stop being billed)
  7. Prints the mAP50 score from the run

Requirements:  pip install requests python-dotenv
"""

import os
import sys
import time
import json
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ── Load config ───────────────────────────────────────────────────────────────

load_dotenv()

VULTR_API_KEY      = os.getenv("VULTR_API_KEY")
VULTR_SSH_KEY_ID   = os.getenv("VULTR_SSH_KEY_ID")
LOCAL_SSH_KEY_PATH = os.path.expanduser(os.getenv("LOCAL_SSH_KEY_PATH", "~/.ssh/id_rsa"))
VULTR_PLAN_ID      = os.getenv("VULTR_PLAN_ID", "vcg-a40-2c-30gb-1x")
VULTR_REGION       = os.getenv("VULTR_REGION", "ewr")

RF_API_KEY         = os.getenv("ROBOFLOW_API_KEY")
RF_WORKSPACE       = os.getenv("ROBOFLOW_WORKSPACE")
RF_PROJECT         = os.getenv("ROBOFLOW_PROJECT")
RF_VERSION         = os.getenv("ROBOFLOW_VERSION", "latest")

YOLO_EPOCHS        = os.getenv("YOLO_EPOCHS", "50")
YOLO_BATCH         = os.getenv("YOLO_BATCH", "16")
YOLO_IMGSZ         = os.getenv("YOLO_IMGSZ", "640")
YOLO_PATIENCE      = os.getenv("YOLO_PATIENCE", "10")
YOLO_MODEL         = os.getenv("YOLO_MODEL", "yolov8n.pt")

LOCAL_OUTPUT_DIR   = Path(os.path.expanduser(os.getenv("LOCAL_OUTPUT_DIR", "~/Downloads")))

VULTR_API_BASE     = "https://api.vultr.com/v2"
VULTR_HEADERS      = {"Authorization": f"Bearer {VULTR_API_KEY}", "Content-Type": "application/json"}

DONE_MARKER        = "/tmp/training_complete"
FAIL_MARKER        = "/tmp/training_failed"


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def check_config():
    missing = []
    for name, val in [
        ("VULTR_API_KEY", VULTR_API_KEY),
        ("VULTR_SSH_KEY_ID", VULTR_SSH_KEY_ID),
        ("ROBOFLOW_API_KEY", RF_API_KEY),
        ("ROBOFLOW_WORKSPACE", RF_WORKSPACE),
        ("ROBOFLOW_PROJECT", RF_PROJECT),
    ]:
        if not val or val.startswith("your_"):
            missing.append(name)
    if missing:
        print(f"\n❌  Missing config values in .env: {', '.join(missing)}")
        print("    Copy .env.example to .env and fill in your values.\n")
        sys.exit(1)

def get_roboflow_version():
    """Returns the numeric version to use (resolves 'latest' automatically)."""
    if RF_VERSION != "latest":
        log(f"Using Roboflow dataset version {RF_VERSION}")
        return RF_VERSION

    log("Looking up latest Roboflow dataset version...")
    url = f"https://api.roboflow.com/{RF_WORKSPACE}/{RF_PROJECT}?api_key={RF_API_KEY}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    versions = data.get("project", {}).get("versions", 0)
    if not versions:
        print("❌  Could not determine latest Roboflow version. Set ROBOFLOW_VERSION manually.")
        sys.exit(1)
    log(f"Latest Roboflow version: {versions}")
    return str(versions)

def build_cloud_init(rf_version):
    """Returns the cloud-init user_data script that runs on the Vultr instance."""
    return f"""#!/bin/bash
set -e

# Set environment variables that cloud-init doesn't provide by default
export HOME=/root
export PATH=$PATH:/usr/local/bin

LOG=/tmp/train.log
exec > >(tee -a $LOG) 2>&1

echo "=== Starting setup ==="

# Install PyTorch with CUDA 12.4 support first, then ultralytics/roboflow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --quiet
pip install ultralytics roboflow --quiet

# Download dataset from Roboflow
echo "=== Downloading dataset (version {rf_version}) ==="
python3 - <<'PYEOF'
from roboflow import Roboflow
rf = Roboflow(api_key="{RF_API_KEY}")
project = rf.workspace("{RF_WORKSPACE}").project("{RF_PROJECT}")
dataset = project.version({rf_version}).download("yolov8", location="/root/dataset")
print("Dataset downloaded to /root/dataset")
PYEOF

# Find the data.yaml
DATA_YAML=$(find /root/dataset -name "data.yaml" | head -1)
echo "Using data.yaml: $DATA_YAML"

# Run training
echo "=== Starting training ==="
python3 - <<PYEOF
from ultralytics import YOLO
import json, glob, os

model = YOLO("{YOLO_MODEL}")
results = model.train(
    data="$DATA_YAML",
    epochs={YOLO_EPOCHS},
    imgsz={YOLO_IMGSZ},
    batch={YOLO_BATCH},
    patience={YOLO_PATIENCE},
    device=0,
    name="chevelle-run",
    exist_ok=True,
)

# Find best.pt and copy to a known location
runs = glob.glob("/root/runs/detect/chevelle-run*/weights/best.pt")
if not runs:
    runs = glob.glob("/root/*.pt")
best = sorted(runs)[-1]
os.makedirs("/root/output", exist_ok=True)

import shutil
shutil.copy(best, "/root/output/best.pt")

# Write results summary
metrics = {{
    "map50":     float(results.results_dict.get("metrics/mAP50(B)", 0)),
    "map50_95":  float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
    "recall":    float(results.results_dict.get("metrics/recall(B)", 0)),
    "model":     best,
}}
with open("/tmp/training_results.json", "w") as f:
    json.dump(metrics, f)

print("Training complete. Results:", metrics)
PYEOF

echo "=== Training finished ==="
touch {DONE_MARKER}
"""

def create_instance(cloud_init_script):
    """Creates the Vultr GPU instance and returns (instance_id, label)."""
    label = f"yolo-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    payload = {
        "region":     VULTR_REGION,
        "plan":       VULTR_PLAN_ID,
        "os_id":      1743,          # Ubuntu 22.04 LTS
        "label":      label,
        "sshkey_id":  [VULTR_SSH_KEY_ID],
        "user_data":  __import__("base64").b64encode(cloud_init_script.encode()).decode(),
        "backups":    "disabled",
    }
    log(f"Creating instance '{label}' ({VULTR_PLAN_ID} in {VULTR_REGION})...")
    r = requests.post(f"{VULTR_API_BASE}/instances", headers=VULTR_HEADERS, json=payload)
    if r.status_code not in (200, 201, 202):
        print(f"❌  Vultr API error {r.status_code}: {r.text}")
        sys.exit(1)
    instance_id = r.json()["instance"]["id"]
    log(f"Instance created: {instance_id}")
    return instance_id, label

def wait_for_instance(instance_id):
    """Polls until the instance is running and has an IP. Returns the IP address."""
    log("Waiting for instance to boot...")
    for _ in range(60):  # up to 10 minutes
        r = requests.get(f"{VULTR_API_BASE}/instances/{instance_id}", headers=VULTR_HEADERS)
        data = r.json()["instance"]
        status = data.get("status")
        power  = data.get("power_status")
        ip     = data.get("main_ip")
        if status == "active" and power == "running" and ip and ip != "0.0.0.0":
            log(f"Instance is up at {ip}")
            return ip
        print(f"  status={status} power={power} ip={ip} — waiting 10s...", end="\r")
        time.sleep(10)
    print("\n❌  Instance never became ready. Check your Vultr dashboard.")
    sys.exit(1)

def wait_for_ssh(ip):
    """Waits until SSH is actually accepting connections."""
    log("Waiting for SSH to be ready...")
    for _ in range(30):
        result = subprocess.run(
            ["ssh", "-i", LOCAL_SSH_KEY_PATH,
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=5",
             "-o", "BatchMode=yes",
             f"root@{ip}", "echo ok"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            log("SSH is ready.")
            return
        time.sleep(10)
    print("❌  SSH never became available.")
    sys.exit(1)

def ssh_poll_for_completion(ip):
    """Polls the instance via SSH until training finishes or fails."""
    log("Training is running on the instance. Polling for completion...")
    log(f"Instance IP: {ip}")
    log(f"Tail logs:   ssh -i {LOCAL_SSH_KEY_PATH} root@{ip} tail -f /tmp/train.log\n")
    start = time.time()
    while True:
        # Check for done or failed markers
        result = subprocess.run(
            ["ssh", "-i", LOCAL_SSH_KEY_PATH,
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10",
             f"root@{ip}",
             f"if [ -f {DONE_MARKER} ]; then echo DONE; elif [ -f {FAIL_MARKER} ]; then echo FAILED; else echo RUNNING; fi"],
            capture_output=True, text=True
        )
        status = result.stdout.strip()
        elapsed = int(time.time() - start)
        print(f"  Training status: {status} ({elapsed}s elapsed)", end="\r")

        if status == "DONE":
            print()
            log("✅  Training complete!")
            return True
        elif status == "FAILED":
            print()
            log("❌  Training failed on the instance. Check logs:")
            subprocess.run(
                ["ssh", "-i", LOCAL_SSH_KEY_PATH, "-o", "StrictHostKeyChecking=no",
                 f"root@{ip}", "tail -50 /tmp/train.log"]
            )
            return False
        time.sleep(30)

def download_results(ip):
    """Downloads best.pt and results JSON from the instance."""
    log(f"Output directory: {LOCAL_OUTPUT_DIR}")
    log(f"Output directory exists: {LOCAL_OUTPUT_DIR.exists()}")
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Output directory after mkdir: {LOCAL_OUTPUT_DIR.exists()}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_model = LOCAL_OUTPUT_DIR / f"best_{timestamp}.pt"
    local_results = LOCAL_OUTPUT_DIR / f"results_{timestamp}.json"
    log(f"Will save model to: {local_model}")

    log(f"Downloading best.pt → {local_model}")
    subprocess.run([
        "scp", "-i", LOCAL_SSH_KEY_PATH,
        "-o", "StrictHostKeyChecking=no",
        f"root@{ip}:/root/output/best.pt",
        str(local_model)
    ], check=True)

    log("Downloading training metrics...")
    subprocess.run([
        "scp", "-i", LOCAL_SSH_KEY_PATH,
        "-o", "StrictHostKeyChecking=no",
        f"root@{ip}:/tmp/training_results.json",
        str(local_results)
    ], check=True)

    with open(local_results) as f:
        metrics = json.load(f)

    return local_model, metrics


def find_existing_instance():
    """Checks Vultr for any already-running GPU instance. Returns (instance_id, ip) or (None, None)."""
    log("Checking for existing Vultr instances...")
    r = requests.get(f"{VULTR_API_BASE}/instances", headers=VULTR_HEADERS)
    if r.status_code != 200:
        log("⚠️  Could not query instances — proceeding to create a new one.")
        return None, None

    instances = r.json().get("instances", [])
    # Filter to GPU instances that are active (not destroyed/pending)
    gpu_instances = [
        i for i in instances
        if i.get("plan", "").startswith("vcg")
        and i.get("status") == "active"
        and i.get("power_status") == "running"
        and i.get("main_ip") not in ("", "0.0.0.0")
    ]

    if not gpu_instances:
        log("No existing GPU instances found — will create a new one.")
        return None, None

    # Use the most recently created one
    instance = gpu_instances[0]
    instance_id = instance["id"]
    ip = instance["main_ip"]
    label = instance.get("label", instance_id)
    log(f"✅  Found existing instance: {label} ({instance_id}) at {ip}")
    return instance_id, ip


def kill_running_training(ip):
    """SSHes into the instance and kills any running training processes."""
    log("Killing any running training processes on the instance...")

    kill_cmd = "; ".join([
        "pkill -f 'python3' || true",
        "pkill -f 'retrain.sh' || true",
        "pkill -f 'ultralytics' || true",
        "kill $(cat /tmp/train.pid 2>/dev/null) 2>/dev/null || true",
        "rm -f /tmp/training_complete /tmp/training_failed /tmp/train.pid /tmp/train.log",
        "echo KILLED"
    ])

    result = subprocess.run(
        ["ssh", "-i", LOCAL_SSH_KEY_PATH,
         "-o", "StrictHostKeyChecking=no",
         "-o", "ConnectTimeout=10",
         f"root@{ip}",
         kill_cmd],
        capture_output=True, text=True
    )

    log(f"  kill stdout: {result.stdout.strip()!r}")
    log(f"  kill stderr: {result.stderr.strip()!r}")
    log(f"  kill returncode: {result.returncode}")

    if result.returncode == 0:
        log("✅  Processes cleared. Instance is ready for a fresh run.")
    else:
        log("⚠️  Kill returned non-zero but continuing anyway.")


def run_training_on_existing(ip, rf_version):
    """Uploads and runs the training script on an already-running instance."""
    log("Uploading training script to existing instance...")

    # Build the training commands as a shell script
    train_script = f"""#!/bin/bash
export HOME=/root
export PATH=$PATH:/usr/local/bin
LOG=/tmp/train.log
exec > >(tee -a $LOG) 2>&1

echo "=== Starting fresh training run ==="

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --quiet
pip install ultralytics roboflow --quiet

echo "=== Downloading dataset (version {rf_version}) ==="
python3 - <<PYEOF
from roboflow import Roboflow
rf = Roboflow(api_key="{RF_API_KEY}")
project = rf.workspace("{RF_WORKSPACE}").project("{RF_PROJECT}")
dataset = project.version({rf_version}).download("yolov8", location="/root/dataset")
print("Dataset downloaded.")
PYEOF

DATA_YAML=$(find /root/dataset -name "data.yaml" | head -1)
echo "Using data.yaml: $DATA_YAML"

echo "=== Starting training ==="
python3 - <<PYEOF
from ultralytics import YOLO
import json, glob, os, shutil

model = YOLO("{YOLO_MODEL}")
results = model.train(
    data="$DATA_YAML",
    epochs={YOLO_EPOCHS},
    imgsz={YOLO_IMGSZ},
    batch={YOLO_BATCH},
    patience={YOLO_PATIENCE},
    device=0,
    name="chevelle-run",
    exist_ok=True,
)

runs = glob.glob("/root/runs/detect/chevelle-run*/weights/best.pt")
best = sorted(runs)[-1]
os.makedirs("/root/output", exist_ok=True)
shutil.copy(best, "/root/output/best.pt")

metrics = {{
    "map50":     float(results.results_dict.get("metrics/mAP50(B)", 0)),
    "map50_95":  float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
    "recall":    float(results.results_dict.get("metrics/recall(B)", 0)),
    "model":     best,
}}
with open("/tmp/training_results.json", "w") as f:
    json.dump(metrics, f)
print("Training complete.", metrics)
PYEOF

echo "=== Done ==="
touch /tmp/training_complete
"""

    # Write script to a temp file and SCP it over
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(train_script)
        tmp_path = f.name

    subprocess.run([
        "scp", "-i", LOCAL_SSH_KEY_PATH,
        "-o", "StrictHostKeyChecking=no",
        tmp_path, f"root@{ip}:/root/retrain.sh"
    ], check=True)
    os.unlink(tmp_path)

    # Run it in the background via nohup so it persists even if SSH drops
    subprocess.run([
        "ssh", "-i", LOCAL_SSH_KEY_PATH,
        "-o", "StrictHostKeyChecking=no",
        f"root@{ip}",
        "chmod +x /root/retrain.sh && nohup /root/retrain.sh > /tmp/train.log 2>&1 & echo $! > /tmp/train.pid"
    ], check=True)
    log("Training started on existing instance.")

def destroy_instance(instance_id):
    """Destroys the Vultr instance to stop billing."""
    log(f"Destroying instance {instance_id}...")
    r = requests.delete(f"{VULTR_API_BASE}/instances/{instance_id}", headers=VULTR_HEADERS)
    if r.status_code in (200, 204):
        log("✅  Instance destroyed. Billing stopped.")
    else:
        log(f"⚠️  Could not destroy instance (status {r.status_code}). Delete it manually in your Vultr dashboard!")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n🚗  YOLO Car Detection — Vultr Training Automation")
    print("=" * 52)

    check_config()

    # Step 1: Resolve Roboflow version
    rf_version = get_roboflow_version()

    # Step 2: Check for existing instance or create a new one
    instance_id, ip = find_existing_instance()

    if instance_id:
        print(f"\n♻️   Reusing existing instance {instance_id} at {ip}")
        cloud_init = None  # Not needed — instance already running
    else:
        print("\n🆕  No existing instance found — creating a new one.")
        cloud_init = build_cloud_init(rf_version)
        instance_id, label = create_instance(cloud_init)

    ip_holder = [None]
    try:
        if not ip:
            # Step 3: Wait for new instance to boot
            ip = wait_for_instance(instance_id)

        ip_holder[0] = ip

        # Step 4: Wait for SSH to be ready
        wait_for_ssh(ip)

        if cloud_init is None:
            # Existing instance — kill stale processes and upload fresh training script
            kill_running_training(ip)
            run_training_on_existing(ip, rf_version)
        # (New instances run training via cloud-init automatically)

        # Step 5: Poll for training completion
        success = ssh_poll_for_completion(ip)

        if success:
            # Step 7: Download results
            model_path, metrics = download_results(ip)

            print("\n" + "=" * 52)
            print("📊  Training Results")
            print(f"    mAP50:      {metrics['map50']:.1%}")
            print(f"    mAP50-95:   {metrics['map50_95']:.1%}")
            print(f"    Precision:  {metrics['precision']:.1%}")
            print(f"    Recall:     {metrics['recall']:.1%}")
            print(f"    Model saved: {model_path}")
            print("=" * 52 + "\n")
        else:
            print("\n⚠️  Training failed — model not downloaded.\n")

    finally:
        # Destroy is commented out so you can SSH in after training to inspect the instance.
        # ⚠️  Remember to manually destroy the instance in your Vultr dashboard when done!
        # destroy_instance(instance_id)
        log(f"⚠️  Instance {instance_id} is still running — destroy it manually in your Vultr dashboard when done.")

    if ip_holder[0]:
        log(f"Done! SSH in with:  ssh -i {LOCAL_SSH_KEY_PATH} root@{ip_holder[0]}")


if __name__ == "__main__":
    main()