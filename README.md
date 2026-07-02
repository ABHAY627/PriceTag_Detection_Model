# Retail Shelf Price Tag Detection & OCR

A complete pipeline that detects price tags on retail shelf images and extracts
numeric price values using object detection (YOLOv8) and OCR (EasyOCR).

---

## Project Structure

```
MUSE_ASSIGNMENT/
├── data/
│   ├── raw/Sample Dataset/     ← 20 sample shelf images
│   ├── splits/                 ← train/val/test split (auto-generated)
│   └── dataset.yaml            ← YOLO dataset config
│
├── scripts/
│   ├── run_inference.py        ← MAIN SCRIPT — run this for demo
│   ├── prepare_dataset.py      ← Step 1: prepare + split dataset
│   ├── augment_dataset.py      ← Step 1: data augmentation
│   ├── generate_labels_cv.py   ← Step 1: bootstrap annotations (CV-based)
│   ├── auto_annotate.py        ← Step 1: bootstrap annotations (YOLO-based)
│   └── train_detector.py       ← Step 2: fine-tune YOLOv8
│
├── backend/
│   └── app/
│       ├── main.py             ← FastAPI app (POST /detect, POST /validate)
│       ├── routers/detect.py   ← API route handlers
│       ├── services/
│       │   ├── detector.py     ← YOLOv8 detection service
│       │   └── ocr_pipeline.py ← OCR + price extraction pipeline
│       └── schemas/detection.py← Pydantic response schemas
│
├── frontend/                   ← Next.js web UI (optional)
├── models/checkpoints/         ← Trained weights saved here (best.pt)
├── results/                    ← Inference output: annotated images + JSON
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. First run downloads EasyOCR model weights (~100MB).

---

## How to Run

### Quick demo — run inference on the sample dataset

```bash
python scripts/run_inference.py --folder "data/raw/Sample Dataset/"
```

This will:
- Detect price tags using classical CV (no trained model needed)
- Run OCR on each detected region
- Save annotated images to `results/`
- Save structured JSON output to `results/`

**Output files:**
```
results/
├── 20240913_161202_1111_2_annotated.jpg   ← image with boxes + prices drawn
├── 20240913_161202_1111_2_result.json     ← structured JSON for this image
├── ...
└── summary_report.json                    ← combined report for all images
```

### Run on a single image

```bash
python scripts/run_inference.py --image "data/raw/Sample Dataset/20240913_161202_1111_2.jpg"
```

### Run with trained YOLO weights (after training)

```bash
python scripts/run_inference.py --folder "data/raw/Sample Dataset/" --weights models/checkpoints/best.pt
```

---

## Full Pipeline (Step by Step)

### Step 1 — Data Preparation

**Generate bootstrap annotations** (classical CV, no annotation tool needed):
```bash
python scripts/generate_labels_cv.py --preview
```
Review preview images in `data/annotation_previews/`, correct in
[LabelImg](https://github.com/HumanSignal/labelImg) if needed.

**Prepare train/val/test split:**
```bash
python scripts/prepare_dataset.py
```

**Augment training data:**
```bash
python scripts/augment_dataset.py --copies 3
```

### Step 2 — Train Detection Model

```bash
# CPU (slower, works on any machine)
python scripts/train_detector.py --model yolov8n --epochs 50 --batch 8

# GPU (recommended)
python scripts/train_detector.py --model yolov8s --epochs 100 --batch 16
```

Saves `best.pt` to `models/checkpoints/`. Writes `DETECTION_REPORT.md` with metrics.

### Step 3 — Run Inference with Trained Model

```bash
python scripts/run_inference.py --folder "data/raw/Sample Dataset/" --weights models/checkpoints/best.pt
```

### Step 4 (Optional) — Run the Web API

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

Endpoints:
- `POST /detect` — upload shelf image → JSON with bounding boxes + prices
- `POST /validate` — cross-check prices against a reference price list
- `GET /health` — health check

### Step 5 (Optional) — Run the Web UI

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## JSON Output Format

Each detected tag produces an entry like this:

```json
{
  "tag_id": 0,
  "bounding_box": {
    "x1": 142.0, "y1": 712.0,
    "x2": 278.0, "y2": 768.0,
    "width": 136.0, "height": 56.0
  },
  "detection_confidence": 0.89,
  "detection_method": "yolov8",
  "price": "2.99",
  "raw_ocr_text": "$2.99",
  "ocr_confidence": 0.91,
  "uncertain": false
}
```

Low-confidence reads are flagged with `"uncertain": true` instead of silently
guessing — these should be reviewed manually.

---

## Detection Performance

Evaluated on the val split after training:

| Metric         | Value  |
|----------------|--------|
| mAP@0.5        | see DETECTION_REPORT.md after training |
| mAP@0.5:0.95   | see DETECTION_REPORT.md after training |

> Note: With only 20 sample images, mAP scores reflect the small dataset size.
> Performance improves significantly with more annotated training data.

---

## OCR Pipeline Design

```
Crop image → CLAHE contrast enhancement → EasyOCR
     ↓ (if confidence < 0.55)
Otsu binarisation → EasyOCR (retry)
     ↓
Regex extraction: [$£€]?\d{1,5}([.,]\d{1,2})?
     ↓
OCR confusion fixes: O→0, B→8, l→1, I→1
     ↓
Sanity check: 0.01 ≤ price ≤ 9999.99
     ↓
Output: price string or "uncertain" flag
```

Google Cloud Vision OCR can be swapped in by setting the `GOOGLE_VISION_KEY`
environment variable — the adapter interface is already in place.

---

## Tech Stack

| Component  | Technology |
|------------|------------|
| Detection  | YOLOv8 (Ultralytics) |
| OCR        | EasyOCR (local, no API key) |
| Backend    | FastAPI + Python |
| Frontend   | Next.js + TypeScript + Tailwind |
| Storage    | JSON files (results/) |

---

## Dataset

20 sample shelf images provided in `data/raw/Sample Dataset/`.
Images are unannotated — bootstrap labels generated via classical CV contour detection.
For production accuracy, manual annotation with LabelImg is recommended.
