"""
Retail Shelf Price Tag Detection & OCR — Main Inference Script
==============================================================
Runs the full pipeline on one or more shelf images:
  1. Detect price tags (YOLOv8 or classical CV fallback)
  2. Crop each detected region
  3. OCR + price extraction (EasyOCR)
  4. Output structured JSON result
  5. Save annotated visualization image

Usage:
    # Single image
    python scripts/run_inference.py --image data/raw/Sample\ Dataset/20240913_161202_1111_2.jpg

    # Entire folder
    python scripts/run_inference.py --folder data/raw/Sample\ Dataset/

    # With trained YOLO weights
    python scripts/run_inference.py --image shelf.jpg --weights models/checkpoints/best.pt

    # Save JSON report
    python scripts/run_inference.py --folder data/raw/Sample\ Dataset/ --output results/

Outputs (saved to --output folder, default = results/):
    results/<image_name>_annotated.jpg   — image with bounding boxes + prices drawn
    results/<image_name>_result.json     — structured JSON per image
    results/summary_report.json          — combined report for all images
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Colours (BGR) ──────────────────────────────────────────────────────────────
COL_BOX       = (0, 196, 245)   # yellow-ish
COL_UNCERTAIN = (0, 60, 230)    # red
COL_LABEL_BG  = (0, 196, 245)
COL_TEXT      = (10, 10, 10)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_with_yolo(img: np.ndarray, weights: Path, conf: float = 0.35) -> list[dict]:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    results = model.predict(img, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": float(box.conf[0]),
                "method": "yolov8",
            })
    return boxes


def detect_with_cv(img: np.ndarray, min_area: float = 0.001, max_area: float = 0.08) -> list[dict]:
    """
    Classical CV fallback — Canny edges + contour filtering.
    Finds small rectangular regions likely to be price tags.
    """
    h, w = img.shape[:2]
    img_area = h * w

    scale  = 640 / w
    small  = cv2.resize(img, (640, int(h * scale)))
    gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area_frac = cv2.contourArea(cnt) / (640 * int(h * scale))
        if not (min_area <= area_frac <= max_area):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < bh or bw / bh < 1.1 or bw / bh > 6.0:
            continue
        # Back to original coords
        x1 = x / scale; y1 = y / scale
        x2 = (x + bw) / scale; y2 = (y + bh) / scale
        boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": 0.60,
            "method": "classical_cv",
        })

    # Remove overlapping boxes
    return _nms(boxes, iou_thresh=0.5)


def _iou(a, b) -> float:
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list, iou_thresh: float = 0.5) -> list:
    boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    kept = []
    for box in boxes:
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — OCR
# ══════════════════════════════════════════════════════════════════════════════

import re

_OCR_FIXES = str.maketrans({"O": "0", "o": "0", "B": "8", "l": "1", "I": "1"})
_PRICE_RE  = re.compile(r"[$£€]?\s*(\d{1,5}(?:[.,]\d{1,2})?)")


def _extract_price(text: str):
    cleaned = text.strip().translate(_OCR_FIXES)
    cleaned = re.sub(r"[^0-9.$£€.,]", " ", cleaned)
    m = _PRICE_RE.search(cleaned)
    if not m:
        return None
    try:
        val = float(m.group(1).replace(",", "."))
        return f"{val:.2f}" if 0.01 <= val <= 9999.99 else None
    except ValueError:
        return None


def _preprocess(img: np.ndarray, aggressive: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    if w < 200:
        img = cv2.resize(img, (200, int(h * 200 / w)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not aggressive:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray  = cv2.fastNlMeansDenoising(gray, h=10)
    else:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


_ocr_reader = None

def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("  📖  Loading EasyOCR model (first time is slow) …")
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def run_ocr(crop: np.ndarray) -> dict:
    """Returns {price, raw_text, confidence, uncertain}"""
    try:
        reader = _get_reader()
    except ImportError:
        return {"price": None, "raw_text": "", "confidence": 0.0,
                "uncertain": True, "note": "easyocr not installed"}

    for aggressive in [False, True]:
        processed = _preprocess(crop.copy(), aggressive)
        results   = reader.readtext(processed, detail=1)
        if not results:
            continue
        results = sorted(results, key=lambda r: r[2], reverse=True)
        for _, text, conf in results:
            price = _extract_price(text)
            if price and conf >= 0.50:
                return {"price": price, "raw_text": text,
                        "confidence": round(float(conf), 4), "uncertain": False}

    # Best guess even if uncertain
    if results:
        _, text, conf = results[0]
        return {"price": _extract_price(text), "raw_text": text,
                "confidence": round(float(conf), 4), "uncertain": True}

    return {"price": None, "raw_text": "", "confidence": 0.0, "uncertain": True}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def draw_results(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        uncertain = det.get("uncertain", False)
        color     = COL_UNCERTAIN if uncertain else COL_BOX
        lw        = 2

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)

        # Corner brackets (scanner viewfinder style)
        cs = max(8, min(x2 - x1, y2 - y1) // 4)
        for (px, py, dx, dy) in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x2, y2, -1, -1), (x1, y2,  1, -1),
        ]:
            cv2.line(out, (px, py), (px + dx * cs, py), color, lw + 1)
            cv2.line(out, (px, py), (px, py + dy * cs), color, lw + 1)

        # Price label
        price = det.get("price") or "?"
        conf  = det.get("detection_confidence", det.get("confidence", 0))
        label = f"${price}  {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        lx, ly = x1, max(y1 - 4, th + 4)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 8, ly + 2), color, -1)
        cv2.putText(out, label, (lx + 4, ly - 2), FONT, 0.45, COL_TEXT, 1, cv2.LINE_AA)

        # Dashed inner box for uncertain
        if uncertain:
            cv2.rectangle(out, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), COL_UNCERTAIN, 1)

    # Summary bar at top
    n_tags    = len(detections)
    n_certain = sum(1 for d in detections if not d.get("uncertain"))
    summary   = f"TAGS: {n_tags}  |  CONFIRMED: {n_certain}  |  UNCERTAIN: {n_tags - n_certain}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(out, summary, (8, 18), FONT, 0.5, (0, 196, 245), 1, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FULL PIPELINE PER IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def process_image(
    img_path: Path,
    weights: Path | None,
    output_dir: Path,
    conf: float = 0.35,
) -> dict:
    print(f"\n📷  Processing: {img_path.name}")
    t0  = time.perf_counter()
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ❌  Cannot read image, skipping.")
        return {}

    h, w = img.shape[:2]

    # --- Detection ---
    if weights and weights.exists():
        print(f"  🔍  Running YOLOv8 detection …")
        boxes = detect_with_yolo(img, weights, conf)
    else:
        print(f"  🔍  Running classical CV detection (no YOLO weights) …")
        boxes = detect_with_cv(img)

    print(f"  ✓   {len(boxes)} tag(s) detected")

    # --- OCR per crop ---
    detections = []
    for i, box in enumerate(boxes):
        x1, y1 = max(0, int(box["x1"])), max(0, int(box["y1"]))
        x2, y2 = min(w, int(box["x2"])), min(h, int(box["y2"]))
        crop    = img[y1:y2, x1:x2]

        if crop.size == 0:
            ocr = {"price": None, "raw_text": "", "confidence": 0.0, "uncertain": True}
        else:
            ocr = run_ocr(crop)

        status = f"${ocr['price']}" if ocr["price"] else "uncertain"
        print(f"    Tag #{i:02d}: {status}  (ocr_conf={ocr['confidence']:.2f})")

        detections.append({
            "tag_id": i,
            "bounding_box": {"x1": box["x1"], "y1": box["y1"],
                             "x2": box["x2"], "y2": box["y2"],
                             "width":  box["x2"] - box["x1"],
                             "height": box["y2"] - box["y1"]},
            "detection_confidence": round(box["confidence"], 4),
            "detection_method":     box["method"],
            "price":                ocr["price"],
            "raw_ocr_text":         ocr["raw_text"],
            "ocr_confidence":       ocr["confidence"],
            "uncertain":            ocr["uncertain"],
        })

    elapsed = round((time.perf_counter() - t0) * 1000, 1)

    # --- Save annotated image ---
    annotated = draw_results(img, detections)
    out_img   = output_dir / f"{img_path.stem}_annotated.jpg"
    cv2.imwrite(str(out_img), annotated)
    print(f"  🖼️   Saved: {out_img.name}")

    # --- Build result dict ---
    result = {
        "image":            img_path.name,
        "image_width":      w,
        "image_height":     h,
        "tag_count":        len(detections),
        "detections":       detections,
        "processing_ms":    elapsed,
        "annotated_image":  str(out_img),
    }

    # --- Save per-image JSON ---
    out_json = output_dir / f"{img_path.stem}_result.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"  📄  Saved: {out_json.name}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Retail Price Tag Detection & OCR — full pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=Path, help="Path to a single shelf image")
    group.add_argument("--folder", type=Path, help="Path to a folder of shelf images")

    parser.add_argument("--weights", type=Path,
                        default=ROOT / "models" / "checkpoints" / "best.pt",
                        help="Path to trained YOLOv8 weights (optional)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results",
                        help="Output folder for annotated images + JSON")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Detection confidence threshold")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Collect images
    if args.image:
        images = [args.image]
    else:
        images = sorted([
            p for p in args.folder.rglob("*")
            if p.suffix.lower() in SUPPORTED_IMG
        ])

    if not images:
        print("❌  No images found.")
        return

    print(f"\n{'='*60}")
    print(f"  PriceTag OCR — processing {len(images)} image(s)")
    print(f"  Weights : {args.weights if args.weights.exists() else 'NOT FOUND — using CV fallback'}")
    print(f"  Output  : {args.output}")
    print(f"{'='*60}")

    all_results = []
    for img_path in images:
        result = process_image(img_path, args.weights, args.output, args.conf)
        if result:
            all_results.append(result)

    # --- Summary report ---
    total_tags     = sum(r["tag_count"] for r in all_results)
    total_certain  = sum(
        sum(1 for d in r["detections"] if not d["uncertain"])
        for r in all_results
    )
    summary = {
        "total_images":    len(all_results),
        "total_tags":      total_tags,
        "confirmed_reads": total_certain,
        "uncertain_reads": total_tags - total_certain,
        "results":         all_results,
    }
    summary_path = args.output / "summary_report.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅  Done.")
    print(f"  Images processed : {len(all_results)}")
    print(f"  Tags detected    : {total_tags}")
    print(f"  Confirmed reads  : {total_certain}")
    print(f"  Uncertain reads  : {total_tags - total_certain}")
    print(f"  Summary JSON     : {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
