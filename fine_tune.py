"""
fine_tune.py – Fine-tune YOLOv8n on the VisDrone MOT4 validation set
(persons only) so the model learns drone-altitude appearance.

Usage:
    python scripts/fine_tune.py --data configs/visdrone_person.yaml --epochs 30

The script:
  1. Downloads / expects the VisDrone dataset at data/VisDrone/
  2. Converts VisDrone annotation format → YOLO format (person class only)
  3. Trains YOLOv8n with drone-optimised hyper-parameters
  4. Exports the best weights to weights/yolov8n_drone.pt
"""

import os
import argparse
import shutil
from pathlib import Path
import yaml

from ultralytics import YOLO


# ─── VisDrone → YOLO annotation converter ────────────────────────────────────
# VisDrone MOT4 annotation columns:
#   frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
#   score, category, truncation, occlusion
#
# Category IDs (1-indexed in VisDrone):
#   1=pedestrian, 2=people, 4=bicycle, 5=car, 6=van, 7=truck,
#   9=tricycle, 10=awning-tricycle, 11=bus, 12=motor
#
# We keep ONLY pedestrian (1) and people (2) → YOLO class 0

PERSON_CATS = {1, 2}   # VisDrone category IDs that map to "person"


def convert_visdrone_to_yolo(visdrone_root: str, out_root: str):
    """
    visdrone_root: path that contains sequences like uav0000009_04358_v/
    Each sequence folder has:
        images/  (jpg files named 0000001.jpg …)
        annotations/  (txt files named 0000001.txt …)
            each annotation line: bbox_left,bbox_top,bbox_width,bbox_height,score,class,truncation,occlusion
    """
    vd = Path(visdrone_root)
    out = Path(out_root)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    sequences = [d for d in vd.iterdir() if d.is_dir()]
    print(f"Converting {len(sequences)} VisDrone sequences …")

    n_images = 0
    n_boxes  = 0

    for seq in sequences:
        img_dir = seq / "img1"        # VisDrone MOT4 uses img1/
        ann_dir = seq / "gt"          # and gt/gt.txt  (MOT format)

        # MOT4 ground truth: single file gt/gt.txt
        gt_file = ann_dir / "gt.txt"
        if not gt_file.exists():
            # Some sequences use per-frame annotations
            ann_dir2 = seq / "annotations"
            if ann_dir2.exists():
                _convert_per_frame(seq, img_dir, ann_dir2, out)
                continue
            else:
                print(f"  [skip] no annotations in {seq.name}")
                continue

        # Parse MOT gt.txt
        frame_boxes: dict = {}
        with open(gt_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 8:
                    continue
                frame_id  = int(parts[0])
                cat       = int(parts[7])
                if cat not in PERSON_CATS:
                    continue
                bx = float(parts[2])
                by = float(parts[3])
                bw = float(parts[4])
                bh = float(parts[5])
                frame_boxes.setdefault(frame_id, []).append((bx, by, bw, bh))

        # Write YOLO label files
        for img_path in sorted(img_dir.glob("*.jpg")):
            frame_id = int(img_path.stem)
            # copy image
            dst_img = out / "images" / f"{seq.name}_{img_path.name}"
            shutil.copy2(img_path, dst_img)

            # get image size
            import cv2
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            ih, iw = im.shape[:2]

            # write labels
            dst_lbl = out / "labels" / f"{seq.name}_{img_path.stem}.txt"
            boxes = frame_boxes.get(frame_id, [])
            with open(dst_lbl, "w") as lf:
                for (bx, by, bw, bh) in boxes:
                    cx = (bx + bw / 2) / iw
                    cy = (by + bh / 2) / ih
                    w  = bw / iw
                    h  = bh / ih
                    lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    n_boxes += 1
            n_images += 1

    print(f"Converted {n_images} images, {n_boxes} person boxes → {out}")


def _convert_per_frame(seq, img_dir, ann_dir, out):
    """Fallback: per-frame .txt annotation files."""
    import cv2
    for img_path in sorted(img_dir.glob("*.jpg")):
        ann_path = ann_dir / (img_path.stem + ".txt")
        dst_img = out / "images" / f"{seq.name}_{img_path.name}"
        shutil.copy2(img_path, dst_img)

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        ih, iw = im.shape[:2]

        dst_lbl = out / "labels" / f"{seq.name}_{img_path.stem}.txt"
        if not ann_path.exists():
            dst_lbl.write_text("")
            continue
        with open(ann_path) as f, open(dst_lbl, "w") as lf:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                cat = int(parts[5])
                if cat not in PERSON_CATS:
                    continue
                bx, by, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                cx = (bx + bw / 2) / iw
                cy = (by + bh / 2) / ih
                w  = bw / iw
                h  = bh / ih
                lf.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ─── YAML dataset config writer ───────────────────────────────────────────────
def write_dataset_yaml(train_path: str, val_path: str, yaml_out: str):
    cfg = {
        "path": ".",
        "train": train_path,
        "val":   val_path,
        "nc":    1,
        "names": ["person"],
    }
    with open(yaml_out, "w") as f:
        yaml.dump(cfg, f)
    print(f"Dataset YAML written to {yaml_out}")


# ─── Training ─────────────────────────────────────────────────────────────────
def train(data_yaml: str, epochs: int, output_dir: str):
    model = YOLO("yolov8n.pt")        # start from COCO pre-trained nano weights

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        workers=4,
        device="cpu",             # change to 0 if you have a GPU
        # ── drone-specific hyper-params ──────────────────────────────────
        lr0=0.005,                # lower LR for fine-tuning
        lrf=0.01,
        warmup_epochs=3,
        mosaic=1.0,               # strong mosaic augmentation helps small objects
        mixup=0.1,
        copy_paste=0.1,           # synthesises more small-object instances
        degrees=10.0,             # allow slight rotation (drone roll)
        scale=0.75,               # scale jitter – drone altitude variation
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        # ── output ───────────────────────────────────────────────────────
        project=output_dir,
        name="drone_person",
        save=True,
        save_period=5,
    )

    best_weights = Path(output_dir) / "drone_person" / "weights" / "best.pt"
    out_weights  = Path("weights") / "yolov8n_drone.pt"
    out_weights.parent.mkdir(exist_ok=True)
    shutil.copy2(best_weights, out_weights)
    print(f"\nBest weights saved to {out_weights}")
    return str(out_weights)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8n on VisDrone persons")
    ap.add_argument("--visdrone-root", default="data/VisDrone/VisDrone2019-MOT-val",
                    help="Root of the VisDrone MOT validation set")
    ap.add_argument("--data",          default="configs/visdrone_person.yaml",
                    help="Output YAML path (will be written by this script)")
    ap.add_argument("--epochs",        type=int, default=30)
    ap.add_argument("--output-dir",    default="runs/train")
    ap.add_argument("--convert-only",  action="store_true",
                    help="Only convert annotations, skip training")
    args = ap.parse_args()

    yolo_data = "data/yolo_visdrone"

    # Step 1 – Convert
    if not Path(yolo_data).exists():
        convert_visdrone_to_yolo(args.visdrone_root, yolo_data)
    else:
        print(f"Converted data already exists at {yolo_data}, skipping conversion.")

    # Step 2 – Write YAML (train = val here since we only have the val split)
    write_dataset_yaml(
        train_path=str(Path(yolo_data) / "images"),
        val_path=str(Path(yolo_data) / "images"),
        yaml_out=args.data,
    )

    if args.convert_only:
        return

    # Step 3 – Train
    train(args.data, args.epochs, args.output_dir)


if __name__ == "__main__":
    main()
