import os
import shutil
from ultralytics import YOLO

# Start with the pretrained YOLO26 nano model
model = YOLO("yolo26n.pt")

# Train on your custom dataset
results = model.train(
    data="/Users/joemcconnell/Documents/Code/yolo-car-test/classic-car-detection/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    patience=10,
    name="chevelle-vs-chevette"
)

# Move best.pt to the same folder as train.py
best_pt = os.path.join(results.save_dir, "weights", "best.pt")
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
shutil.copy(best_pt, dest)
print(f"Model saved to: {dest}")