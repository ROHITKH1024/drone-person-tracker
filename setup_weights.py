#!/usr/bin/env python3
"""
setup_weights.py – Downloads YOLOv8n base weights if not present.
Fine-tuned weights must be produced by running scripts/fine_tune.py.
"""
from pathlib import Path
from ultralytics import YOLO

weights_dir = Path("weights")
weights_dir.mkdir(exist_ok=True)

base_pt = weights_dir / "yolov8n.pt"
drone_pt = weights_dir / "yolov8n_drone.pt"

# Download base weights (auto-cached by ultralytics)
print("Downloading/verifying YOLOv8n base weights …")
model = YOLO("yolov8n.pt")
import shutil
src = Path("yolov8n.pt")
if src.exists() and not base_pt.exists():
    shutil.move(str(src), str(base_pt))
elif not base_pt.exists():
    import urllib.request
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    print(f"Fetching {url} …")
    urllib.request.urlretrieve(url, str(base_pt))

# If fine-tuned weights don't exist, copy base as placeholder
if not drone_pt.exists():
    shutil.copy2(base_pt, drone_pt)
    print(f"Placeholder weights copied to {drone_pt}")
    print("Run 'python scripts/fine_tune.py' to produce drone-specific weights.")
else:
    print(f"Drone weights already at {drone_pt}")

print("\nSetup complete.")
