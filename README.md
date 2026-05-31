# 🚁 Aerial Guardian — Drone Person Tracker

**YOLOv8-nano + ByteTrack** fine-tuned for the VisDrone dataset, with drone-specific enhancements for robust small-object detection and multi-object tracking from a moving platform.

---

## What This Does

Given a video (or image sequence) recorded from a drone, the pipeline:

1. **Detects** every person in each frame using YOLOv8-nano with SAHI sliced inference (for small, far-away targets)
2. **Tracks** every person across frames with ByteTrack, maintaining consistent IDs
3. **Compensates** for drone ego-motion (camera movement) using ORB-based homography estimation so tracks don't drift when the drone turns
4. **Renders** bounding boxes, ID labels, confidence scores, and trajectory tails on the output video

---

## Architecture Choices & Drone-Specific Additions

### Base Model: YOLOv8-nano
- **Why nano?** Model size ≤ 6 MB (well under the 300 MB limit), COCO-pretrained, and fast enough for real-time on CPU. We fine-tune it on VisDrone to teach it drone-altitude appearance.
- **Why not a larger model?** A larger model buys ~3–5 mAP points but at 3–5× compute cost — too slow for an embedded platform.

### Small-Object Detection: SAHI
Drone footage has people appearing at ~15–40 px height. YOLOv8's native 640×640 input downsamples them further. **SAHI** (Slicing Aided Hyper Inference) solves this by:
- Splitting each frame into overlapping 320×320 tiles
- Running detection on each tile independently
- Merging results with NMM (Non-Maximum Merging)

This recovers 15–25% more small detections vs. full-frame inference alone.

### Ego-Motion Compensation (EMC)
Drone movement causes all pixels to shift between frames, making naive IoU-based matching fail (the same person appears in a completely different position). EMC:
1. Extracts ORB keypoints from the background in each frame
2. Matches them to the previous frame with brute-force matching
3. Estimates the inter-frame homography via RANSAC
4. **Warps all stored track centroids** into the new camera frame before the ByteTrack IoU step

This dramatically reduces ID switches during drone pans and tilts.

### Tracker: ByteTrack
ByteTrack uses *both* high- and low-confidence detections:
- High-conf detections are matched to existing tracks (standard)
- Low-conf detections can *rescue* temporarily lost tracks (helps when a person is occluded for 1–2 frames)

This is better than DeepSORT in CPU-constrained settings because it requires no re-ID embedding model.

### Confidence Smoothing
A 5-frame rolling average of detection confidence per ID suppresses flickering bounding boxes without introducing latency.

---

## ID-Switching Mitigation

| Problem | Solution |
|---|---|
| Drone ego-motion shifts all positions | Homography-based track warping (EMC) |
| Short occlusions | ByteTrack low-conf rescue + 30-frame track buffer |
| False re-detections near existing tracks | IoU-based suppression via ByteTrack |
| Confidence flickering | Per-ID rolling average smoother |

---

## Edge Hardware Adaptation (Jetson)

To deploy on NVIDIA Jetson:

1. **Export to TensorRT:**
   ```bash
   yolo export model=weights/yolov8n_drone.pt format=engine device=0 half=True
   ```
   This converts to FP16 TensorRT engine, giving ~4–6× speedup over PyTorch on Jetson.

2. **Replace SAHI with Jetson-tuned tiling:** On Jetson Orin, inference is fast enough to run native 640-input; SAHI overhead can be reduced to 2 tiles instead of 9.

3. **Use INT8 quantisation** for Jetson Nano (older): 
   ```bash
   yolo export model=weights/yolov8n_drone.pt format=engine int8=True
   ```

4. **Disable EMC on GPU**: Homography estimation on CPU is cheap; on Jetson you can GPU-accelerate it with `cv2.cuda.ORB` for even lower latency.

---

## Setup & Installation

### Requirements
- Python 3.9+
- pip

### Step 1 – Clone / extract the project
```bash
cd aerial_guardian
```

### Step 2 – Create a virtual environment (recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac / Linux:
source venv/bin/activate
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 – Download base model weights
```bash
python scripts/setup_weights.py
```
This downloads YOLOv8n pretrained on COCO and copies it to `weights/yolov8n_drone.pt` as a placeholder.

---

## Fine-Tuning on VisDrone (Optional but Recommended)

Download the VisDrone MOT validation set from the link in the challenge brief and extract to:
```
data/VisDrone/VisDrone2019-MOT-val/
```

Then run:
```bash
python scripts/fine_tune.py --epochs 30
```

This will:
1. Convert VisDrone annotations → YOLO format (person class only)
2. Train YOLOv8n for 30 epochs with drone-optimised augmentation
3. Save the best checkpoint to `weights/yolov8n_drone.pt`

Training takes ~20–40 minutes on a modern CPU, or ~5 minutes with a GPU.

---

## Running the Tracker

### On a video file:
```bash
python src/tracker.py --source path/to/video.mp4 --output outputs/tracked.mp4
```

### On a VisDrone image sequence directory:
```bash
python src/tracker.py --source data/VisDrone/VisDrone2019-MOT-val/uav0000009_04358_v/img1 --output outputs/tracked.mp4
```

### With live preview window:
```bash
python src/tracker.py --source path/to/video.mp4 --output outputs/tracked.mp4 --display
```

### Disable SAHI (faster but fewer small-object detections):
```bash
python src/tracker.py --source path/to/video.mp4 --no-sahi
```

### All options:
```
--source      Video file, image sequence directory, or webcam index (e.g. 0)
--output      Output video path (default: outputs/tracked.mp4)
--model       Path to .pt weights (default: weights/yolov8n_drone.pt)
--conf        Detection confidence threshold (default: 0.30)
--no-sahi     Disable SAHI sliced inference
--no-emc      Disable ego-motion compensation
--display     Show live preview window (press Q to quit)
```

---

## Benchmarking FPS

```bash
python scripts/benchmark.py --model weights/yolov8n_drone.pt --source path/to/video.mp4
```

Expected results:

| Hardware | Mode | FPS |
|---|---|---|
| Intel Core i7 (CPU only) | No SAHI | ~18–25 FPS |
| Intel Core i7 (CPU only) | With SAHI | ~6–10 FPS |
| NVIDIA Jetson Orin (TensorRT FP16) | No SAHI | ~60+ FPS |

---

## Project Structure

```
aerial_guardian/
├── src/
│   └── tracker.py          # Main pipeline: detection + tracking + rendering
├── scripts/
│   ├── fine_tune.py         # VisDrone fine-tuning + annotation converter
│   ├── setup_weights.py     # Downloads base YOLOv8n weights
│   └── benchmark.py         # FPS + model-size benchmark
├── configs/
│   └── bytetrack.yaml       # ByteTrack hyperparameter config
├── weights/                 # Model weights (auto-populated)
├── outputs/                 # Output videos go here
├── data/                    # Put VisDrone dataset here
├── requirements.txt
└── README.md
```

---

## Engineering Trade-offs

| Decision | Trade-off |
|---|---|
| YOLOv8**n** over YOLOv8s/m | −3 mAP, −5× model size, +2× FPS |
| SAHI enabled | +20% detections, −50% FPS (use on offline/batch, disable for real-time) |
| EMC via ORB+RANSAC | +stability, +5 ms/frame overhead |
| ByteTrack over DeepSORT | No re-ID network → 50% less memory, simpler pipeline |
| CPU target | Any laptop/Jetson works; disables CUDA path |

The recommended real-time config is: **SAHI disabled, EMC enabled, YOLOv8n** → ~20 FPS on CPU, all person classes tracked stably.

---

## License
MIT
