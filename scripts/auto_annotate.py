"""
Checkpoint 1B — Auto-Annotation Bootstrap
==========================================
Since the dataset is unannotated, this script uses a pretrained YOLOv8 model
to detect rectangular objects (likely price tags) and generates initial
YOLO-format .txt label files.

Strategy:
  - Use YOLOv8n pretrained on COCO (detects generic objects)
  - Filter for small rectangular detections (likely tags/labels)
  - All detections get class_id=0 (price_tag)
  - Output: .txt files in data/raw/Sample Dataset/

After running this, manually review bounding boxes in LabelImg or Roboflow,
then proceed with prepare_dataset.py.

Usage:
    python scripts/auto_annotate.py [--conf 0.25] [--iou 0.45]
"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    import cv2
except ImportError:
    print("❌  Missing dependencies. Run:")
    print("    pip install ultralytics opencv-python")
    exit(1)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "Sample Dataset"

SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_yolo_label(path: Path, boxes: list) -> None:
    """boxes = list of [cx, cy, w, h] normalized 0-1."""
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
    path.write_text("\n".join(lines))


def is_likely_tag(w_norm: float, h_norm: float) -> bool:
    """Heuristic: price tags are usually small rectangles."""
    # Typical tag: 5-20% of image width, 3-15% of image height
    return (0.05 <= w_norm <= 0.25) and (0.03 <= h_norm <= 0.20)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate using YOLOv8 pretrained model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"❌  {RAW_DIR} not found.")
        return

    imgs = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in SUPPORTED_IMG])
    if not imgs:
        print(f"❌  No images in {RAW_DIR}")
        return

    print(f"🔍  Loading YOLOv8n (COCO pretrained) …")
    model = YOLO("yolov8n.pt")  # downloads automatically if missing

    annotated_count = 0
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Could not read {img_path.name}, skipping.")
            continue

        h_img, w_img = img.shape[:2]
        results = model.predict(img_path, conf=args.conf, iou=args.iou, verbose=False)

        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = ((x1 + x2) / 2) / w_img
                cy = ((y1 + y2) / 2) / h_img
                w = (x2 - x1) / w_img
                h = (y2 - y1) / h_img

                # Filter: keep only small rectangular objects (likely tags)
                if is_likely_tag(w, h):
                    boxes.append([cx, cy, w, h])

        if boxes:
            label_path = img_path.with_suffix(".txt")
            write_yolo_label(label_path, boxes)
            annotated_count += 1
            print(f"  ✓  {img_path.name} → {len(boxes)} tag(s)")
        else:
            print(f"  ⚠️  {img_path.name} → no tags detected")

    print(f"\n✅  Auto-annotated {annotated_count} / {len(imgs)} images.")
    print("    Review/correct labels in LabelImg, then run:")
    print("      python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
