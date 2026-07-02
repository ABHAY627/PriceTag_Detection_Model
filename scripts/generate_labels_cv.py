"""
Checkpoint 1B — Classical CV Label Bootstrap
=============================================
Uses OpenCV edge detection + contour analysis to find rectangular regions
that look like price tags (small rectangles, high contrast against shelf background).

This is a BOOTSTRAP — generates candidate labels for human review.
Review with LabelImg before trusting them for training.

Usage:
    python scripts/generate_labels_cv.py [--min-area 0.002] [--max-area 0.06]
"""

import argparse
import shutil
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("❌  Run: pip install opencv-python numpy")
    exit(1)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "Sample Dataset"
SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detect_tag_regions(img: np.ndarray, min_area_frac: float, max_area_frac: float) -> list:
    """
    Returns list of (cx_norm, cy_norm, w_norm, h_norm) for candidate tag regions.
    Strategy:
      1. Resize to 640px wide for speed
      2. Convert to LAB colour space — tags often have uniform colour
      3. Canny edge detection
      4. Find contours, filter for rectangular-ish shapes within area range
    """
    h_orig, w_orig = img.shape[:2]
    scale = 640 / w_orig
    w_s = 640
    h_s = int(h_orig * scale)
    small = cv2.resize(img, (w_s, h_s))
    img_area = w_s * h_s

    # --- Edge detection ---
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    # Dilate edges to close small gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # --- Contour analysis ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_frac = area / img_area

        if not (min_area_frac <= area_frac <= max_area_frac):
            continue

        # Fit bounding rect
        x, y, w, h = cv2.boundingRect(cnt)

        # Aspect ratio: price tags are wider than tall, 1.2x to 5x
        if w < h:
            continue
        aspect = w / h
        if not (1.1 <= aspect <= 6.0):
            continue

        # Rectangularity check: contour area vs bbox area
        rect_area = w * h
        if rect_area == 0 or area / rect_area < 0.35:
            continue

        # Convert back to original image coordinates & normalise
        cx_norm = ((x + w / 2) / w_s)
        cy_norm = ((y + h / 2) / h_s)
        w_norm = w / w_s
        h_norm = h / h_s
        candidates.append((cx_norm, cy_norm, w_norm, h_norm))

    # Deduplicate heavily overlapping candidates (simple IoU check)
    deduped = []
    for a in candidates:
        skip = False
        for b in deduped:
            if _iou(a, b) > 0.5:
                skip = True
                break
        if not skip:
            deduped.append(a)

    return deduped


def _iou(a, b) -> float:
    ax1 = a[0] - a[2] / 2; ay1 = a[1] - a[3] / 2
    ax2 = a[0] + a[2] / 2; ay2 = a[1] + a[3] / 2
    bx1 = b[0] - b[2] / 2; by1 = b[1] - b[3] / 2
    bx2 = b[0] + b[2] / 2; by2 = b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def write_label(path: Path, boxes: list) -> None:
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
    path.write_text("\n".join(lines))


def visualise(img_path: Path, boxes: list, out_dir: Path) -> None:
    """Save a debug image with boxes drawn on it."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    out_path = out_dir / f"preview_{img_path.name}"
    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-area", type=float, default=0.002,
                        help="Min tag area as fraction of image")
    parser.add_argument("--max-area", type=float, default=0.06,
                        help="Max tag area as fraction of image")
    parser.add_argument("--preview", action="store_true",
                        help="Save preview images with bounding boxes drawn")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"❌  {RAW_DIR} not found.")
        return

    imgs = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in SUPPORTED_IMG])
    if not imgs:
        print(f"❌  No images found in {RAW_DIR}")
        return

    preview_dir = ROOT / "data" / "annotation_previews"
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍  Analysing {len(imgs)} images with classical CV …\n")

    total_boxes = 0
    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Cannot read {img_path.name}")
            continue

        boxes = detect_tag_regions(img, args.min_area, args.max_area)
        lbl_path = img_path.with_suffix(".txt")
        write_label(lbl_path, boxes)
        total_boxes += len(boxes)
        status = f"{len(boxes)} box(es)"
        print(f"  {'✓' if boxes else '⚠'}  {img_path.name:45s} → {status}")

        if args.preview:
            visualise(img_path, boxes, preview_dir)

    print(f"\n✅  Generated {total_boxes} candidate annotations across {len(imgs)} images.")
    if args.preview:
        print(f"    Preview images saved → data/annotation_previews/")
    print("\n⚠️   IMPORTANT: These are bootstrap labels — review with LabelImg before training.")
    print("    LabelImg: https://github.com/HumanSignal/labelImg")
    print("\n    Next: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
