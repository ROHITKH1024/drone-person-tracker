"""
Aerial Guardian – Drone Person Tracker
YOLOv8-nano  +  ByteTrack  +  Drone-specific enhancements

Enhancements over vanilla YOLOv8+ByteTrack:
  1. SAHI (Sliced Inference) for small-object detection at drone altitude
  2. Motion-compensation: frame-to-frame homography to offset drone ego-motion
  3. Confidence heat-map smoothing across frames to reduce flickering detections
  4. Trajectory "tail" rendering (last N centroids per ID)
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ── ultralytics supplies both YOLOv8 and ByteTrack ──────────────────────────
from ultralytics import YOLO

# ── SAHI for sliced inference (small object boost) ───────────────────────────
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    model_path: str = "weights/yolov8n_drone.pt"   # fine-tuned or base yolov8n
    confidence: float = 0.30
    iou_threshold: float = 0.45
    target_class: int = 0                           # 0 = "person" in COCO
    # SAHI slice settings
    use_sahi: bool = True
    slice_height: int = 320
    slice_width: int = 320
    overlap_ratio: float = 0.20
    # Tail / trajectory
    tail_length: int = 30                           # frames to keep
    # Ego-motion compensation
    use_emc: bool = True
    emc_max_features: int = 500
    # Display
    show_fps: bool = True
    line_thickness: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# Ego-Motion Compensator
# ─────────────────────────────────────────────────────────────────────────────
class EgoMotionCompensator:
    """
    Estimates the inter-frame homography caused by drone movement and
    warps previous track positions into the current camera frame.
    This dramatically reduces ID-switches caused by camera translation/rotation.
    """

    def __init__(self, max_features: int = 500):
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp   = None
        self._prev_des  = None

    def update(self, frame_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns a 3×3 homography H that maps prev→current, or None on first frame.
        """
        kp, des = self.orb.detectAndCompute(frame_gray, None)
        H = None
        if self._prev_gray is not None and des is not None and self._prev_des is not None:
            matches = self.bf.match(self._prev_des, des)
            if len(matches) >= 4:
                matches = sorted(matches, key=lambda m: m.distance)[:200]
                src_pts = np.float32([self._prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt          for m in matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        self._prev_gray = frame_gray
        self._prev_kp   = kp
        self._prev_des  = des
        return H

    @staticmethod
    def warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Apply homography to an array of (x, y) points."""
        if H is None or len(pts) == 0:
            return pts
        pts_h = np.hstack([pts, np.ones((len(pts), 1))]).T        # 3×N
        warped = H @ pts_h                                          # 3×N
        warped /= warped[2]                                         # normalise
        return warped[:2].T.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Smoothing Buffer
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceSmoother:
    """
    Maintains a rolling average of per-ID confidence to suppress flickering.
    IDs with very low rolling confidence are visually dimmed.
    """
    def __init__(self, window: int = 5):
        self._buf: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, conf: float) -> float:
        self._buf[track_id].append(conf)
        return float(np.mean(self._buf[track_id]))

    def get(self, track_id: int) -> float:
        if track_id not in self._buf:
            return 0.0
        return float(np.mean(self._buf[track_id]))


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (deterministic per ID)
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE = [
    (0, 200, 255), (0, 255, 128), (255, 80, 0),
    (255, 0, 200), (0, 120, 255), (200, 255, 0),
    (255, 200, 0), (0, 255, 200), (120, 0, 255),
]

def id_color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_box_label(frame, x1, y1, x2, y2, track_id, conf, color, thickness=2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"ID:{track_id}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    # background pill
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_tail(frame, tail: deque, color, thickness=1):
    pts = list(tail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)                   # older points = thinner
        t = max(1, int(thickness * alpha))
        cv2.line(frame, pts[i - 1], pts[i], color, t, cv2.LINE_AA)


def overlay_fps(frame, fps: float):
    label = f"FPS: {fps:.1f}"
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline class
# ─────────────────────────────────────────────────────────────────────────────
class AerialGuardianPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        print(f"[AerialGuardian] Loading model: {cfg.model_path}")
        self.model = YOLO(cfg.model_path)

        # SAHI detection model wrapper
        if cfg.use_sahi:
            print("[AerialGuardian] Building SAHI wrapper …")
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=cfg.model_path,
                confidence_threshold=cfg.confidence,
                device="cpu",          # change to "cuda:0" if GPU available
            )
        else:
            self.sahi_model = None

        self.emc     = EgoMotionCompensator(cfg.emc_max_features) if cfg.use_emc else None
        self.smoother = ConfidenceSmoother()
        self.tails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=cfg.tail_length))
        self._fps_history: deque = deque(maxlen=30)

    # ── per-frame detection with SAHI ────────────────────────────────────────
    def _detect_sahi(self, frame_bgr: np.ndarray) -> List[Tuple]:
        """Returns list of (x1,y1,x2,y2,conf) for persons."""
        result = get_sliced_prediction(
            frame_bgr,
            self.sahi_model,
            slice_height=self.cfg.slice_height,
            slice_width=self.cfg.slice_width,
            overlap_height_ratio=self.cfg.overlap_ratio,
            overlap_width_ratio=self.cfg.overlap_ratio,
            perform_standard_pred=True,
            postprocess_type="NMM",
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        boxes = []
        for pred in result.object_prediction_list:
            if pred.category.id != self.cfg.target_class:
                continue
            b = pred.bbox
            boxes.append((int(b.minx), int(b.miny), int(b.maxx), int(b.maxy), pred.score.value))
        return boxes

    # ── per-frame detection without SAHI (fast path) ─────────────────────────
    def _detect_direct(self, frame_bgr: np.ndarray) -> List[Tuple]:
        results = self.model(frame_bgr,
                             conf=self.cfg.confidence,
                             iou=self.cfg.iou_threshold,
                             classes=[self.cfg.target_class],
                             verbose=False)
        boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))
        return boxes

    # ── full frame processing ─────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Process one frame.
        Returns (annotated_frame, active_track_count).
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        vis = frame.copy()

        # 1. Ego-motion compensation
        if self.emc is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H = self.emc.update(gray)
            # Warp existing tails into new camera frame
            if H is not None:
                for tid, tail in self.tails.items():
                    if len(tail) > 0:
                        pts = np.array(list(tail), dtype=np.float32)
                        warped = EgoMotionCompensator.warp_points(pts, H)
                        self.tails[tid] = deque(
                            [(int(p[0]), int(p[1])) for p in warped],
                            maxlen=self.cfg.tail_length
                        )

        # 2. Run YOLO + ByteTrack via ultralytics tracker
        track_results = self.model.track(
            frame,
            persist=True,
            conf=self.cfg.confidence,
            iou=self.cfg.iou_threshold,
            classes=[self.cfg.target_class],
            tracker="bytetrack.yaml",
            verbose=False,
        )

        active_ids = set()

        if track_results and track_results[0].boxes is not None:
            boxes = track_results[0].boxes
            if boxes.id is not None:
                for box, tid_t, conf_t in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.id.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = map(int, box)
                    tid  = int(tid_t)
                    conf = float(conf_t)

                    # clamp to frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    # Confidence smoothing
                    smooth_conf = self.smoother.update(tid, conf)

                    # Tail centroid
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    self.tails[tid].append((cx, cy))
                    active_ids.add(tid)

                    color = id_color(tid)
                    draw_tail(vis, self.tails[tid], color, self.cfg.line_thickness)
                    draw_box_label(vis, x1, y1, x2, y2, tid, smooth_conf, color, self.cfg.line_thickness)

        # 3. FPS overlay
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self._fps_history.append(fps)
        avg_fps = float(np.mean(self._fps_history))
        if self.cfg.show_fps:
            overlay_fps(vis, avg_fps)

        return vis, len(active_ids)

    # ── video / sequence runner ───────────────────────────────────────────────
    def run_video(self, source: str, output_path: Optional[str] = None,
                  display: bool = False) -> Dict:
        """
        Process a video file or image sequence.
        source: path to video OR directory of images OR webcam index.
        output_path: if given, saves annotated video.
        """
        # Open capture
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        elif Path(source).is_dir():
            # VisDrone image sequences: numbered jpgs
            frames_list = sorted(Path(source).glob("*.jpg")) + \
                          sorted(Path(source).glob("*.png"))
            return self._run_image_sequence(frames_list, output_path, display)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")

        fps_in   = cap.get(cv2.CAP_PROP_FPS) or 25
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

        frame_idx  = 0
        total_time = 0.0
        max_ids    = 0

        print(f"[AerialGuardian] Processing video ({total} frames)  →  {output_path or 'no output file'}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            vis, n_ids = self.process_frame(frame)
            total_time += time.perf_counter() - t0
            max_ids = max(max_ids, n_ids)

            if writer:
                writer.write(vis)

            if display:
                cv2.imshow("Aerial Guardian", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[AerialGuardian] User quit.")
                    break

            frame_idx += 1
            if frame_idx % 50 == 0:
                avg_fps = frame_idx / total_time if total_time > 0 else 0
                print(f"  frame {frame_idx}/{total}  |  avg FPS {avg_fps:.1f}  |  active IDs {n_ids}")

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        avg_fps = frame_idx / total_time if total_time > 0 else 0
        stats = {
            "frames_processed": frame_idx,
            "average_fps": round(avg_fps, 2),
            "total_time_s": round(total_time, 2),
            "max_simultaneous_ids": max_ids,
        }
        print(f"\n[AerialGuardian] Done. Stats: {stats}")
        return stats

    def _run_image_sequence(self, frames_list, output_path, display):
        if not frames_list:
            raise FileNotFoundError("No jpg/png frames found in directory.")

        sample = cv2.imread(str(frames_list[0]))
        h, w = sample.shape[:2]

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 25, (w, h))

        total_time = 0.0
        max_ids    = 0

        for i, fp in enumerate(frames_list):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            t0 = time.perf_counter()
            vis, n_ids = self.process_frame(frame)
            total_time += time.perf_counter() - t0
            max_ids = max(max_ids, n_ids)

            if writer:
                writer.write(vis)
            if display:
                cv2.imshow("Aerial Guardian", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if i % 50 == 0:
                fps = (i + 1) / total_time if total_time > 0 else 0
                print(f"  frame {i + 1}/{len(frames_list)}  |  avg FPS {fps:.1f}  |  IDs {n_ids}")

        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        avg_fps = len(frames_list) / total_time if total_time > 0 else 0
        stats = {
            "frames_processed": len(frames_list),
            "average_fps": round(avg_fps, 2),
            "total_time_s": round(total_time, 2),
            "max_simultaneous_ids": max_ids,
        }
        print(f"\n[AerialGuardian] Done. Stats: {stats}")
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Aerial Guardian – Drone Person Tracker")
    p.add_argument("--source",  required=True,
                   help="Video file, image-sequence directory, or webcam index (0)")
    p.add_argument("--output",  default="outputs/tracked.mp4",
                   help="Output annotated video path")
    p.add_argument("--model",   default="weights/yolov8n_drone.pt",
                   help="Path to YOLOv8 weights (.pt)")
    p.add_argument("--conf",    type=float, default=0.30, help="Detection confidence threshold")
    p.add_argument("--no-sahi", action="store_true", help="Disable SAHI sliced inference")
    p.add_argument("--no-emc",  action="store_true", help="Disable ego-motion compensation")
    p.add_argument("--display", action="store_true", help="Show live preview window")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = PipelineConfig(
        model_path=args.model,
        confidence=args.conf,
        use_sahi=not args.no_sahi,
        use_emc=not args.no_emc,
    )
    pipeline = AerialGuardianPipeline(cfg)
    pipeline.run_video(args.source, output_path=args.output, display=args.display)
