"""
benchmark.py – Measures FPS and prints model size.

Usage:
    python scripts/benchmark.py --model weights/yolov8n_drone.pt --source data/sample.mp4
"""
import time
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def run_benchmark(model_path: str, source: str, n_frames: int = 200):
    model = YOLO(model_path)
    size  = model_size_mb(model_path)
    print(f"Model : {model_path}")
    print(f"Size  : {size:.1f} MB  ({'✓ OK' if size < 300 else '✗ OVER LIMIT'})")

    cap = cv2.VideoCapture(source) if not Path(source).is_dir() else None

    times = []
    for i in range(n_frames):
        if cap:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        else:
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        model.track(frame, persist=True, conf=0.3, classes=[0],
                    tracker="bytetrack.yaml", verbose=False)
        times.append(time.perf_counter() - t0)

    if cap:
        cap.release()

    avg   = sum(times) / len(times)
    fps   = 1.0 / avg
    p95   = sorted(times)[int(0.95 * len(times))]

    print(f"\nResults over {len(times)} frames:")
    print(f"  Avg FPS    : {fps:.1f}")
    print(f"  Avg latency: {avg * 1000:.1f} ms")
    print(f"  P95 latency: {p95 * 1000:.1f} ms")
    print(f"  Hardware   : CPU (no GPU)")
    return fps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="weights/yolov8n_drone.pt")
    ap.add_argument("--source", default="0",
                    help="Video file, image dir, or '0' for synthetic benchmark")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()
    run_benchmark(args.model, args.source, args.frames)
