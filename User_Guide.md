# YOLO Car Detection — Training Pipeline Guide

This document covers the full workflow for improving your Chevelle/Chevette detection model: adding new images in Roboflow, training on Vultr, and managing your cloud instance.

---

## Overview

The pipeline works in three steps:

1. **Add & label images** in Roboflow
2. **Train the model** on a Vultr GPU instance
3. **Manage the instance** to avoid unnecessary charges

---

## Step 1: Adding Images to Roboflow

### Upload new images

1. Go to [app.roboflow.com](https://app.roboflow.com) and open your project
2. Click **Upload Data** in the left sidebar
3. Drag and drop your new images or click to browse — JPG, PNG, and WEBP are all supported
4. Click **Save and Continue**

### Label the images

1. After uploading, Roboflow will take you to the annotation queue
2. Click an image to open the annotation editor
3. Use the **Bounding Box** tool to draw a box around the car
4. Assign the correct class — `chevelle-1972` or `chevette-1983`
5. Click **Save** and move to the next image
6. Repeat until all new images are labeled

> **Tip:** Use the Smart Polygon tool to auto-detect the car outline, then switch to bounding box mode. Look for a red outline around the car (not a full red fill) — that's the bounding box style YOLO expects.

### Generate a new dataset version

Once all images are labeled:

1. Click **Versions** in the left sidebar
2. Click **Generate New Version**
3. Set your train/valid/test split (70/20/10 is recommended)
4. Under **Preprocessing**, keep resize at 640x640
5. Optionally enable augmentations — **Horizontal Flip** is the most useful
6. Click **Generate**

Roboflow combines all your previous images with the new ones automatically. Your old versions are preserved so you can always compare results.

---

## Step 2: Training on Vultr

### Prerequisites

Make sure your `.env` file is filled in with your Vultr and Roboflow credentials. See `.env.example` for all required values.

Your virtual environment should be active before running any scripts:

```bash
source venv/bin/activate
```

Confirm it's active — your prompt should show `(venv)` at the start.

### Run the training script

```bash
python3 train_on_vultr.py
```

The script will:

- Check if a Vultr GPU instance is already running and reuse it, or create a new one
- Pull your latest Roboflow dataset version automatically
- Run YOLO training on the GPU
- Download `best.pt` to your local models folder
- Print your results when complete

### What to expect

**If creating a new instance:**
```
🆕  No existing instance found — creating a new one.
[10:32:14] Creating instance 'yolo-train-20260424-103214'...
[10:32:14] Waiting for instance to boot...
[10:32:45] Instance is up at 45.63.22.120
```

**If reusing an existing instance:**
```
♻️   Reusing existing instance abc-123 at 45.63.22.120
[10:32:14] Killing any running training processes on the instance...
[10:32:15] ✅  Processes cleared. Instance is ready for a fresh run.
```

**While training is running**, you can watch live progress in a second terminal:
```bash
ssh -i ~/.ssh/id_rsa root@<ip> tail -f /tmp/train.log
```

The IP address is printed in the output once the instance is up.

**When training completes**, you'll see results like:
```
📊  Training Results
    mAP50:      95.7%
    mAP50-95:   86.5%
    Precision:  87.4%
    Recall:     95.2%
    Model saved: ~/Documents/Code/yolo-car-test/models/best_20260424_221501.pt
```

### Understanding the results

| Metric | What it means |
|---|---|
| **mAP50** | Overall accuracy — above 90% is excellent |
| **mAP50-95** | Stricter accuracy across multiple thresholds |
| **Precision** | When it says "Chevelle", how often it's correct |
| **Recall** | How many actual cars it successfully finds |

### Testing the new model locally

After training, run your test script to verify the new model on your local test images:

```bash
python3 test_model.py
```

The test script automatically picks up the most recently trained model from your models folder. Compare confidence scores against previous runs in your tracking spreadsheet.

---

## Step 3: Managing Your Vultr Instance

> ⚠️ Vultr charges by the hour. Always check and terminate your instance when you're done to avoid unnecessary charges.

### Check if an instance is running

```bash
python3 check_instance.py
```

This shows all running GPU instances, how long they've been up, and the estimated cost so far:

```
Label                          IP                 Uptime             Est. Cost
--------------------------------------------------------------------------------
yolo-train-20260424-221501     45.63.22.120       2h 14m 32s         $0.2652
```

### Terminate an instance

```bash
python3 terminate_instance.py
```

This will list any running instances and ask you to confirm before terminating:

```
Found 1 GPU instance(s):

  [1] yolo-train-20260424-221501  |  45.63.22.120  |  f6b235a0-...

Options:
  Enter a number to terminate that instance
  Enter 'q' to quit without terminating

Your choice: 1
Terminating yolo-train-20260424-221501... ✅  Done. Billing stopped.
```

### SSH into the instance directly

If you need to inspect the instance manually:

```bash
ssh -i ~/.ssh/id_rsa root@<ip>
```

The IP is printed when `train_on_vultr.py` starts. You can also find it in your [Vultr dashboard](https://my.vultr.com).

---

## Full Workflow Summary

```
1. Add images to Roboflow → label them → generate new version
2. python3 train_on_vultr.py
3. Review results printed in terminal
4. python3 test_model.py  (optional local test)
5. python3 terminate_instance.py  (when done)
```

---

## Configuration Reference

All settings live in your `.env` file. Key values:

| Variable | Description |
|---|---|
| `VULTR_API_KEY` | Your Vultr API key (Settings → API) |
| `VULTR_SSH_KEY_ID` | SSH key ID registered in Vultr |
| `VULTR_PLAN_ID` | GPU plan — `vcg-a16-2c-16g-4vram` recommended |
| `VULTR_REGION` | Region code — `ewr` is New Jersey |
| `ROBOFLOW_API_KEY` | Your Roboflow private API key |
| `ROBOFLOW_WORKSPACE` | Workspace slug from your Roboflow URL |
| `ROBOFLOW_PROJECT` | Project slug from your Roboflow URL |
| `ROBOFLOW_VERSION` | Dataset version — `latest` always uses newest |
| `YOLO_EPOCHS` | Max training epochs (default: 50) |
| `YOLO_BATCH` | Images per training step (default: 32) |
| `LOCAL_OUTPUT_DIR` | Where `best.pt` is saved on your Mac |

---

## Troubleshooting

**API key errors from Vultr**
Make sure there are no spaces around the `=` in your `.env` file and no leading spaces before the variable name.

**Training fails with CUDA error**
The script installs PyTorch compiled for CUDA 12.4 to match the Vultr instance drivers. If you see a CUDA version error, the instance may have been created before the fix — terminate it and let the script create a fresh one.

**Model not downloading to Mac**
Check that `LOCAL_OUTPUT_DIR` in your `.env` has no leading spaces and points to a valid path. The script creates the folder automatically if it doesn't exist.

**Instance not found after restarting the script**
If you terminated the instance manually or it was destroyed, the script will automatically create a new one on the next run.